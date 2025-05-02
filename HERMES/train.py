import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split

from model import *
from dataset import *

data_dir = '/glade/derecho/scratch/aidenz/data/HERMES_data/'
label_csv_path = '/glade/derecho/scratch/aidenz/data/HERMES_labels/sampled_parameters.csv'

batch_size = 64

# Get the dataset then split it into training and validation sets
dataset = spectraDataset(label_csv_path, data_dir, transform=ToTensor())
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(13))

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '13313'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare_ddp(rank, world_size, batch_size=batch_size, pin_memory=False, num_workers=0):
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=val_sampler)
    
    return train_dataloader, val_dataloader

def cleanup_ddp():
    dist.destroy_process_group()

def main(rank, world_size):

    # Setup DDP
    setup_ddp(rank, world_size)

    # Prepare DDP dataloaders
    train_dataloader, val_dataloader = prepare_ddp(rank, world_size)

    # Create the model
    model = SpecGen(in_channels=13, out_channels=44228).to(rank)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Define the loss function and optimizer
    L_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20)

    # Add physical loss terms
    # Transit depth loss
    
