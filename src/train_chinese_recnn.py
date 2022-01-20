import os
import torch

from tqdm.auto import tqdm
from transformers import get_scheduler
from model import ChineseDictNet
from dataset import build_xinhua_dataloader


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

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    model = ChineseDictNet()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_dataloader, val_dataloader = build_xinhua_dataloader(batch_size,
                                                               num_workers=0)
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
        for step, (batch, word_ids) in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(word_ids=word_ids, **batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                p_bar.set_description(f"loss: {loss}")
        p_bar.update(1)
        # # validation
        # model.eval()
        # res = {}
        # for batch in val_dataloader:
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     pred_embed = model(**batch)
        #     res[int(batch['word_ids'].item())] = pred_embed.detach().cpu()
        # # calculate embeddings
        # bert_norm = model.embedding_weight[list(res.keys())].norm(dim=1)
        # # %%
        # recnn_norm = torch.cat(list(res.values())).norm(dim=1)
        # # %%
        # acc = sum(bert_norm > recnn_norm) / 100
        # p_bar.set_description(f"acc: {acc:.4f}")
        # p_bar.update(1)


def main():
    config = dict(batch_size=8, norm_range=(0, 1.35), epochs=200, lr=1e-3)
    train(config)


if __name__ == "__main__":
    main()