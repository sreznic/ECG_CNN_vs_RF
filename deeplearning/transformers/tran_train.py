import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

from utils.functions import read_config
import pandas as pd
import numpy as np
from tran_dataset import TorchECGDatasetDataDict
from tran_model import CTN, CNN
from tran_noam_opt import NoamOpt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score as sklearn_f1_score
from tran_train_utils import get_datasets, get_augm_datasets, start_log, write_log
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

DIR_PATH = "deeplearning/transformers"

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers
deepfeat_sz = 64
dropout_rate = 0.2

def train(epoch, model, dataset, optimizer, leads, weights, device):
    model.train()
    losses, probs, labels = [], [], []
    for i, (x, y) in enumerate(dataset):
        x = x[:, leads, :].to(device).float()
        y = y.to(device).float()
        optimizer.zero_grad()
        out = model(x)
        # loss = F.binary_cross_entropy_with_logits(out, y)
        loss = F.binary_cross_entropy_with_logits(out, y, weight=weights)
        # loss = CrossEntropyLoss(weight=weights)(out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probs.append(out.sigmoid().detach().cpu())
        labels.append(y.detach().cpu())
    loss = np.mean(losses)

    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    f1_score = sklearn_f1_score(torch.argmax(labels, axis=1), torch.argmax(probs, axis=1), average='macro')
    return loss, f1_score

def validate(epoch, model, dataset, leads, device):
    model.eval()
    losses, probs, labels = [], [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(dataset):
            x = x[:, leads, :].to(device).float()
            y = y.to(device).float()
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y)

            losses.append(loss.item())
            probs.append(out.sigmoid().detach().cpu())
            labels.append(y.detach().cpu())
    loss = np.mean(losses)

    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    f1_score = sklearn_f1_score(torch.argmax(labels, axis=1), torch.argmax(probs, axis=1), average='macro')
    return loss, f1_score

def main(config, debug):
    leads = config['leads']

    sample_records_train_each = 5000
    sample_records_val_test_total = 1000
    max_transformations = 3#10
    record_len = 3000
    num_epochs = config['epoch_num']
    if debug:
        sample_records_train_each = 2000
        sample_records_val_test_total = 400
        max_transformations = 0
        num_epochs = config['debug_epoch_num']
    
    train_dataset, val_dataset, _, train_weights = get_augm_datasets(
        config, config['arrhythmia_label'], sample_records_train_each, sample_records_val_test_total,\
        max_transformations, record_len)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_weights = train_weights.to(device)
    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, [i for i in range(3)]).to(device)
    # model = CNN().to(device)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # opt = NoamOpt(256, 1, 10, torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    start_log(config['log_path'], config['arrhythmia_label'], leads)

    for epoch in range(num_epochs):
        train_loss, train_f1_score = train(epoch, model, train_dataset, opt, leads, train_weights, device)
        val_loss, val_f1_score = validate(epoch, model, val_dataset, leads, device)
        print(f'Epoch {epoch + 1}/{num_epochs}: Train loss: {train_loss:.3f}, ' +\
            f'Train F1 score: {train_f1_score:.3f}, Val loss: {val_loss:.3f}, ' +\
            f'Val F1 score: {val_f1_score:.3f}')
        write_log(config['log_path'], config['arrhythmia_label'], leads,\
                  epoch, train_loss, train_f1_score, val_loss, val_f1_score)
        pass
    pass

if __name__ == '__main__':
    config = read_config(f'{DIR_PATH}/train_config.json')
    main(config, True)