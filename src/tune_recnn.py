import os
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from model import DictNet
from dataset import build_dataset
from transformers import DataCollatorWithPadding
from argparse import ArgumentParser
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter


def save_model(model, loss, path='./'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, f'{path}/recnn-{loss}.pt')


def train(config, checkpoint_dir=None):

    # Hyper Parameters
    norm_range = config['norm_range']
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['epochs']

    tokenized_dataset, tokenizer = build_dataset(range=norm_range)
    val_dataset = tokenized_dataset.shuffle(seed=42)['train'].select(
        range(100))

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

        # # validation
        # model.eval()
        # res = {}
        # for batch in val_dataloader:
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     pred_embed = model(**batch)['pred_embed']
        #     res[int(batch['word_ids'].item())] = pred_embed.detach().cpu()
        # # calculate embeddings
        # bert_norm = model.embedding_weight[list(res.keys())].norm(dim=1)
        # # %%
        # recnn_norm = torch.cat(list(res.values())).norm(dim=1)
        # # %%
        # acc = sum(bert_norm > recnn_norm) / 100

        tune.report(train_loss=loss.cpu().detach().numpy())

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)


def main():
    reporter = CLIReporter(max_progress_rows=10)
    reporter.add_metric_column("train_loss")
    scheduler = ASHAScheduler(max_t=100, grace_period=1, reduction_factor=2)
    analysis = tune.run(train,
                        fail_fast=True,
                        metric='train_loss',
                        mode='min',
                        num_samples=-1,
                        scheduler=scheduler,
                        resources_per_trial={'gpu': 0.5},
                        config={
                            'lr':
                            tune.grid_search([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
                            'batch_size': tune.choice([8]),
                            'epochs': tune.choice([30, 60, 100]),
                            'norm_range': (0, 1.35)
                        })


if __name__ == "__main__":
    main()