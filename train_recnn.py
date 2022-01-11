import os
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from model import DictNet
from dataset import build_dataset
from transformers import DataCollatorWithPadding
from argparse import ArgumentParser


def save_model(model, loss, path='./'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, f'{path}/recnn-{loss}.pt')


def train(config, checkpoint_dir=None):
    """
    train Firt step to train recnn.

    Recnn required two steps for training, this function is for the first one.

    Args:
        config (dict): dictionary containing norm_range, batch_size, lr, epochs.
        checkpoint_dir (str, optional): checkpoint point derectory, if you want to resume training.
    """

    # Hyper Parameters
    norm_range = config['norm_range']
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['epochs']

    tokenized_dataset, tokenizer = build_dataset(range=norm_range)
    tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    val_dataset = tokenized_dataset['train'].select(range(100))

    collate_function = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(tokenized_dataset['train'],
                                  shuffle=False,
                                  batch_size=batch_size,
                                  collate_fn=collate_function,
                                  num_workers=8)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                batch_size=1,
                                collate_fn=collate_function,
                                num_workers=8)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    model = DictNet()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_train_step = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler('linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_train_step)

    # Train
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    p_bar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # validation
        model.eval()
        res = {}
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_embed = model(**batch)
            res[int(batch['word_ids'].item())] = pred_embed.detach().cpu()
        # calculate embeddings
        bert_norm = model.embedding_weight[list(res.keys())].norm(dim=1)
        # %%
        recnn_norm = torch.cat(list(res.values())).norm(dim=1)
        # %%
        acc = sum(bert_norm > recnn_norm) / 100
        p_bar.set_description(f"acc: {acc:.4f}")
        p_bar.update(1)


def main():
    config = dict(batch_size=8, norm_range=(0, 1.35), epochs=10, lr=1e-3)
    train(config)


if __name__ == "__main__":
    main()