import json
import os
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet101, resnet50, inception_v3, resnet34, resnet18

from dataset import StretcherDataset
import statistics


class Net(pl.LightningModule):
    def __init__(self, root, batch_size):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.root = root

        self.spec_net = nn.Sequential(*list(resnet50(pretrained=False).children())[:-1], nn.ReLU(True))
        self.image_net = nn.Sequential(*list(resnet50(pretrained=False).children())[:-1], nn.ReLU(True))

        self.fc_loc = nn.Sequential(
            nn.Linear(4096, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        s = x['spec'][:, None, :, :]
        spec = torch.cat((s, s, s), 1)
        i = x['image'][:, None, :, :]
        image = torch.cat((i, i, i), 1)

        spec_out = self.spec_net(spec)
        spec_out = spec_out.view(-1, 2048)

        img_out = self.image_net(image.float())
        img_out = img_out.view(-1, 2048)

        combined_out = torch.cat((spec_out, img_out), 1)
        midpoint = self.fc_loc(combined_out)
        # print(midpoint)

        image_width = x['image'].shape[1]
        map = x['map'][19][:, None]
        map = map / image_width - 0.5

        loss = F.mse_loss(midpoint, map, reduction='mean')
        accuracy = 1 / loss.mean().item()

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        losses, accuracy = self.forward(batch)
        tensorboard_logs = {
            'train_loss': losses,
            'train_accuracy': accuracy
        }
        return {'loss': losses, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        losses, accuracy = self.forward(batch)
        return {'val_loss': losses, 'val_acc': accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = statistics.mean([x['val_acc'] for x in outputs])
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0003)

    @pl.data_loader
    def train_dataloader(self):
        dataset = StretcherDataset(
            root=self.root + '/train'
        )
        dist_sampler = torch.utils.data.RandomSampler(dataset)  # torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=self.batch_size,
        )

    @pl.data_loader
    def val_dataloader(self):
        dataset = StretcherDataset(
            root=self.root + '/val'
        )
        dist_sampler =  torch.utils.data.RandomSampler(dataset) # torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=self.batch_size,
        )
