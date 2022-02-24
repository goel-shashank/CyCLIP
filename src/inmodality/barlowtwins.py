import requests
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import lightly
from transformers import CLIPVisionModel, CLIPVisionConfig
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from PIL import Image


'''
Requirements

pip install lightly
pip install transformers
'''
parser = argparse.ArgumentParser(description = 'Barlow Twins Training')

parser.add_argument('--train_images_csv', type = str, help = 'csv which contains about training images')
parser.add_argument('--validation_images_csv', type = str, help = 'csv which contains about validation images')
parser.add_argument('--image_key', type = str, default = 'image', help= 'column name in csv containing image names')
parser.add_argument("--delimiter", type = str, default = ",", help = "For train/validation data csv file, the delimiter to use")
parser.add_argument("--epochs", type = int, default = 1, help = "Number of train epochs")
parser.add_argument("--batch_size", type = int, default = 32, help = "batch size")

parser.add_argument('--checkpoint_file', type = str, help = 'location of checkpoint')

args = parser.parse_args()


class ImageDataset(Dataset):

  def __init__(self, path, image_key, delimiter):

      df = pd.read_csv(path, sep = delimiter)
      self.dir = os.path.dirname(path)
      self.images = df[image_key].tolist()

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    image = Image.open(os.path.join(self.dir, self.images[idx])).convert('RGB')
    return (image, 0, 'image.png')

class ProjectionHead(nn.Module):
  
    def __init__(self, blocks: List[Tuple[int, int, nn.Module, nn.Module]]):

        super(ProjectionHead, self).__init__()

        self.layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            self.layers.append(nn.Linear(input_dim, output_dim))
            if batch_norm:
                self.layers.append(batch_norm)
            if non_linearity:
                self.layers.append(non_linearity)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.
        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        return self.layers(x)

class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.
    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]
    [0]: 2021, Barlow Twins, https://arxiv.org/abs/2103.03230
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(BarlowTwinsProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])

class BT(nn.Module):

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 768,
                 proj_hidden_dim: int = 768,#8192,
                 out_dim: int = 768 #8192
                ):

        super(BT, self).__init__()

        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = BarlowTwinsProjectionHead(
            num_ftrs,
            proj_hidden_dim,
            out_dim
        )

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
                
        # forward pass first input
        f0 = self.backbone(x0).pooler_output
        out0 = self.projection_mlp(f0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0

        # forward pass second input
        f1 = self.backbone(x1).pooler_output
        out1 = self.projection_mlp(f1)

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        return out0, out1

class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):

        device = z_a.device
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D, device = device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss

class BarlowTwins(pl.LightningModule):
    def __init__(self):
        super(BarlowTwins, self).__init__()
        
        self.clip_vision = CLIPVisionModel(CLIPVisionConfig())
        self.clip_bt = BT(self.clip_vision)
        self.criterion = BarlowTwinsLoss()
            
    def forward(self, x):
        return self.clip_bt(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.clip_bt(x0, x1)
        z_a = x0
        z_b = x1
        loss = self.criterion(z_a, z_b)
        self.log('train_loss_ssl', loss)
        return loss

    # learning rate warm-up
    def optimizer_steps(self,
                        epoch=None,
                        batch_idx=None,
                        optimizer=None,
                        optimizer_idx=None,
                        optimizer_closure=None,
                        on_tpu=None,
                        using_native_amp=None,
                        using_lbfgs=None):        
        # 120 steps ~ 1 epoch
        if self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 1e-3

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.clip_bt.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

def return_pretrained_clip_vision():

    model = BarlowTwins()

    gpus = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if gpus else 'cpu'

    cpt = torch.load(args.checkpoint_file, map_location = device)
    model.load_state_dict(cpt['state_dict'])
    
    return model.clip_bt.backbone

if __name__ == '__main__':
    
    num_workers = 8
    max_epochs = args.epochs
    batch_size = args.batch_size
    seed=1

    pl.seed_everything(seed)

    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=224,
        gaussian_blur=0.,
    )

    gpus = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if gpus else 'cpu'

    train_dataset = ImageDataset(args.train_images_csv, args.image_key, args.delimiter)
    # validation_dataset = ImageDataset(args.validation_images_csv, args.image_key, args.delimiter)

    dataloader_train_ssl = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            drop_last=True,
                            num_workers=num_workers
                        )
    # dataloader_validation_ssl = torch.utils.data.DataLoader(
    #                         validation_dataset,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         collate_fn=collate_fn,
    #                         drop_last=True,
    #                         num_workers=num_workers
    #                     )
    
    model = BarlowTwins()

    trainer = pl.Trainer(max_epochs=max_epochs, 
                         gpus=1,
                         progress_bar_refresh_rate=100, 
                         log_every_n_steps = 10,
                        )
    
    trainer.fit(
        model,
        dataloader_train_ssl,
    )


