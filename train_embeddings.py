import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from model import BertRecNN
from dataset import build_dataset, build_embeddings_dataset
from transformers import DataCollatorWithPadding


def train(config):
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
    for epoch in range(config['epochs']):
        for input, target_position in tqdm(dataloader):
            input = {i: v.to(device) for i, v in input.items()}
            last_hidden_state = model(input)['last_hidden_state']


def main():
    config = dict(batch_size=8, norm_range=(0, 1.35), epochs=10, lr=1e-3)
    train(config)


if __name__ == "__main__":
    main()