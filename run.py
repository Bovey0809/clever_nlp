from os import environ
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from model import DictNet
from dataloader import build_dataset
from transformers import DataCollatorWithPadding
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


def save_model(model, loss):
    torch.save(model, f'recnn-{loss}.pt')


def setup(rank, world_size):
    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = '12345'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo(rank, world_size):
    setup(rank, world_size)

    # Build Dataset and tokenizer
    tokenized_dataset, tokenizer = build_dataset(name='wordnet_nltk')
    collate_function = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(tokenized_dataset['train'],
                                  shuffle=False,
                                  batch_size=256,
                                  collate_fn=collate_function,
                                  num_workers=8)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    model = DictNet()
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-2)
    num_epochs = 300

    num_train_step = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler('linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_train_step)

    # Train
    progress_bar = tqdm(range(num_train_step))
    model.train()
    min_loss = float('inf')
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = ddp_model(**batch)
            loss = outputs['loss']
            loss.backward()
            if rank == 0 and loss < min_loss:
                min_loss = loss
                save_model(model, 'best')
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_description(f"loss: {loss:.5f}")
            progress_bar.update(1)
        if rank == 0:
            save_model(model, str(epoch) + "epoch")
    if rank == 0:
        save_model(model, "last")
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo, args=(world_size, ), nprocs=world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    run_demo(demo, world_size)