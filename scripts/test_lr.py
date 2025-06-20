#-*- coding: utf-8 -*-
from __future__ import absolute_import
import time

import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse
from datetime import datetime
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

from configs.get_config import load_config
from models import *
from datasets import *
from losses import *
from lib.core_function import validate, train, test
from logs.logger import Logger, LOG_DIR
from lib.optimizers.sam import SAM
from lib.scheduler.linear_decay import LinearDecayLR

from matplotlib import pyplot as plt

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def args_parser(args=None):
    parser = argparse.ArgumentParser("Training process...")
    parser.add_argument('--cfg', help='Config file', required=True)
    parser.add_argument('--alloc_mem', '-a',  help='Pre allocating GPU memory', action='store_true')
    return parser.parse_args(args)

def test_and_plot_lr_scheduler(n_epochs=100, start_decay=30, base_lr=0.01, booster=5):
    """
    Tests the LinearDecayLR scheduler and plots its behavior over epochs.
    """
    # 1. Setup a dummy model parameter and optimizer
    dummy_model_param = torch.nn.Parameter(torch.randn(1))
    optimizer = optim.Adam([dummy_model_param], lr=cfg.TRAIN.lr, weight_decay=2e-5)
    for param_group in optimizer.param_groups:
      if 'initial_lr' not in param_group:
        param_group['initial_lr'] = param_group['lr']
    
    # 2. Instantiate the scheduler
    scheduler = LinearDecayLR(optimizer, n_epochs, n_epochs//4, last_epoch=0, booster=4)

    # 3. Simulate a training loop to save LR values
    lr_history = []
    print("Simulating training loop...")
    for epoch in range(n_epochs):
        # Get the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: Learning Rate = {current_lr:.6f}")
        
        # Advance the scheduler to the next step
        scheduler.step()

    # 4. Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_epochs + 1), lr_history, marker='o', linestyle='-', markersize=4)
    
    # Add a vertical line to show where the decay starts
    plt.axvline(x=n_epochs//4, color='r', linestyle='--', label=f'Decay Starts (Epoch {n_epochs//4})')
    
    plt.title('Learning Rate Schedule Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, n_epochs + 1)
    plt.ylim(0) # Ensure y-axis starts at 0
    plt.savefig('mon_graphique.png')

# --- Script Entry Point ---
if __name__ == '__main__':
    if len(sys.argv[1:]):
        args = sys.argv[1:]
    else:
        args = None
    
    args = args_parser(args)
    cfg = load_config(args.cfg)

    test_and_plot_lr_scheduler(
        n_epochs=cfg.TRAIN.epochs,
        base_lr=cfg.TRAIN.lr,
        booster=4
    )