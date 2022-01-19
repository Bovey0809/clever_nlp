import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from model import DictNet
import os
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from model import DictNet
from dataset import build_dataset
from transformers import DataCollatorWithPadding


class LitRecnn(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dict_net = DictNet()

    def forward(self, x):
        outputs = self.dict_net(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = self.dict_net(**train_batch)
        self.log('train_loss', loss['loss'])
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.dict_net.forward_train(**val_batch)
        self.log('val_loss', loss['loss'])


if __name__ == '__main__':
    config = dict(batch_size=8, norm_range=(0, 1.35), epochs=200, lr=1e-3)
    # data
    norm_range = config['norm_range']
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['epochs']

    tokenized_dataset, tokenizer = build_dataset(range=norm_range)
    tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    val_dataset = tokenized_dataset['train'].select(range(100))

    collate_function = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(tokenized_dataset['train'],
                              shuffle=False,
                              batch_size=batch_size,
                              collate_fn=collate_function,
                              num_workers=8)
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=1,
                            collate_fn=collate_function,
                            num_workers=8)

    # model
    model = LitRecnn()

    # training
    trainer = pl.Trainer(gpus=1,
                         num_nodes=1,
                         fast_dev_run=False,
                        #  overfit_batches=True,
                        #  auto_scale_batch_size=True,
                         auto_lr_find=True)
    trainer.fit(model, train_loader, val_loader)
