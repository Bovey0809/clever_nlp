import torch
import pickle
from datetime import datetime
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from model import BertRecNN
from dataset import build_embeddings_dataset
from transformers import DataCollatorWithPadding


def train(config):
    with open(config['recnn_embeddings_sheet'], 'rb') as f:
        recnn_word_embeddings_dict = pickle.load(f)

    model = BertRecNN()
    dataset, tokenizer = build_embeddings_dataset(config['norm_range'])
    dataset = dataset.shuffle(seed=42)
    padding_fn = DataCollatorWithPadding(tokenizer, padding=True)

    def _collate_fn(batch):
        target_indexes = []
        for i in batch:
            target_index = i.pop('target_word_position')
            target_indexes.append(torch.tensor(target_index))
        res = padding_fn(batch)
        return res, target_indexes

    dataloader = DataLoader(dataset['train'],
                            batch_size=config['batch_size'],
                            collate_fn=_collate_fn,
                            shuffle=False)

    # Move to CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        running_loss = 0.0
        epoch_steps = 0

        for i, (input, target_position) in tqdm(enumerate(dataloader, 0)):
            input = {i: v.to(device) for i, v in input.items()}
            optimizer.zero_grad()

            bert_output_vectors, word_ids = model(input, target_position)

            # Calculate loss
            recnn_embeddings = []
            for word_id in word_ids:
                tokens = tokenizer.convert_ids_to_tokens(word_id)
                for token in tokens:
                    recnn_embedding = recnn_word_embeddings_dict[token]
                    recnn_embeddings.append(recnn_embedding)

            recnn_embeddings = torch.tensor(recnn_embeddings,
                                            device=bert_output_vectors.device)
            loss = mse_loss(recnn_embeddings, bert_output_vectors)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1

            if i % 200 == 199:
                print("[%d, %5d] loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        embeddings = model.get_input_embeddings()
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y-%H-%M-%S")
        torch.save(embeddings, f"recnn_stage2_epoch_{epoch}_{dt_string}.pt")


def main():
    config = dict(batch_size=8,
                  norm_range=(1.35, 2),
                  epochs=20,
                  lr=1e-3,
                  recnn_embeddings_sheet="./words_preembeddings_dict.p")
    train(config)


if __name__ == "__main__":
    main()