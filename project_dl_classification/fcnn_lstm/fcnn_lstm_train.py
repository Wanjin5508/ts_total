import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# import torch.optim as optim
# from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
# import os
sys.path.append('../')

# from train import train_one_epoch, validate
# from dataset import PartialDischargeDataset, custom_collate_with_lengths, custom_collate
from utils import EarlyStopping

print(torch.__version__)
print(torch.cuda.is_available())


def train_epoch(model, loader, optimizer, criterion, device='cuda'):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y, lengths in loader:
        # print(f'x: {x.size()}')
        # print(f'y: {y.size()}')
        x, y = x.to(device), y.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(x, lengths)
        # print(f'train one epoch, logits size: {logits.size()}, y size: {y.size()}')
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device='cuda'):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y, lengths in loader:
        # print(f'x: {x.size()}')
        # print(f'y: {y.size()}')
        x, y = x.to(device), y.to(device)
        lengths = lengths.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total



