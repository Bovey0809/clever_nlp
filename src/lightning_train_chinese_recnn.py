import torch

import pytorch_lightning as pl
from model import ChineseDictNet

from dataset import build_xinhua_dataloader


class LitRecnn(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.dict_net = ChineseDictNet()

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
    train_dataloader, val_dataloader = build_xinhua_dataloader(batch_size,
                                                               num_workers=0)

    # model
    model = LitRecnn()

    # training
    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        fast_dev_run=False,
        #  overfit_batches=True,
        #  auto_scale_batch_size=True,
        auto_lr_find=True)
    trainer.fit(model, train_dataloader, val_dataloader)
