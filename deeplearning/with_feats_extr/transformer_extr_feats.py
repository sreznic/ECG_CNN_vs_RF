import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

import argparse
import os
from tqdm import tqdm
import numpy as np
import sklearn.metrics
from scipy.stats import ttest_ind
import keras.models
import gc
import time
from sklearn.utils import class_weight
from utils.snomed_ct import get_snomed
from dataset import Dataset, split_dataset_custom
from utils.functions import read_config, get_dataset_df, get_all_records
import datahelpers.dldata as dldata
import math
from keras.utils import Sequence
from transformer_network import CTN, TestArch
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from biosppy.signals.tools import filter_signal
from utils.functions import resample_to_500hz
from transformer_optimizer import NoamOpt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def apply_filter(signal, filter_bandwidth = [3, 45], fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth, 
                                     sampling_rate=fs)
        return signal

def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

DIR_PATH = "deeplearning/with_feats_extr"

def get_dataset(dataframe, config, sequence_len, target_labels):
    dataset = dldata.DLDataset(dataframe, config['dataset_path'], record_len=sequence_len)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

def get_pred(dataset, model):
    y_score = model.predict(dataset, verbose=1)

    json_pred = {
        "pred": y_score,
        "true": dataset.get_all_ys()
    }

    return json_pred

def get_datasets(config, leads):
    def make_record(record):
        # record = apply_filter(record)
        # record = normalize(record)
        return record
    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])
    test_df = get_dataset_df(config, config['arrhythmia_label'], config['test_description_file_name'])
    target_labels = [config['arrhythmia_label'], 'NSR', 'Other']
    
    train_dataset = get_dataset(train_df, config, config['record_len'], target_labels)
    val_dataset = get_dataset(val_df, config, config['record_len'], target_labels)
    test_dataset = get_dataset(test_df, config, config['record_len'], target_labels)

    train_dataset.set_limit_batches(config['train_batches'])
    train_dataset.balance_samples_in_classes()
    val_dataset.set_limit_batches(config['val_batches'])
    test_dataset.set_limit_batches(config['test_batches'])
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.shuffle_df()
        dataset.set_leads(leads)
        # dataset.should_resample = True
        dataset.set_custom_make_record(make_record)
    _ = train_dataset[0]

    return train_dataset, val_dataset, test_dataset

def get_whole_dataset(config, leads):
    records_names = get_all_records(config, config['whole_description_file_name'])
    records_names = records_names
    whole_df = pd.DataFrame({
        'Record': records_names,
        'Arrhythmia': ['Dummy'] * len(records_names)
    })
    dl_dataset = DLDatasetNoBatch(whole_df, config['dataset_path'], config['record_len'])
    dl_dataset.set_transform_y(lambda _: [0, 0, 0])
    dl_dataset.set_leads(leads)
    data = np.array([dl_dataset[i][0] for i in range(len(dl_dataset))])
    return data, records_names

def train(epoch, model, train_dataset, optimizer, target_labels):
    model.train()
    class_weights = train_dataset.get_class_weights(target_labels)
    class_weights = [class_weights[0], class_weights[1], class_weights[2]]
    class_weights = torch.FloatTensor(class_weights).to(device)
    class_weights = None
    losses, probs, labels_list = [], [], []
    for i, (records_and_labels) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        records = torch.from_numpy(records_and_labels[0]).float().to(device)
        labels = torch.from_numpy(records_and_labels[1]).float().to(device)
        records = records.transpose(1, 2)
        optimizer.optimizer.zero_grad()
        out = model(records)
        if class_weights is None:
            loss = F.binary_cross_entropy_with_logits(out, labels)
        else:
            loss = F.binary_cross_entropy_with_logits(out, labels, weight=class_weights)
        loss.backward()
        optimizer.step()

        prob = out.sigmoid().data.cpu().numpy()
        losses.append(loss.item())
        probs.append(prob)
        labels_list.append(labels.data.cpu().numpy())

    loss = np.mean(losses)
    probs = np.concatenate(probs)

    f1_score = sklearn.metrics.f1_score(
        np.argmax(np.concatenate(labels_list), axis=1), 
        np.argmax(probs, axis=1), 
        average='macro')

    return loss, f1_score

def validate(epoch, model, dataset, optimizer, target_labels):
    model.eval()
    class_weights = dataset.get_class_weights(target_labels)
    class_weights = [class_weights[0], class_weights[1], class_weights[2]]
    class_weights = torch.FloatTensor(class_weights).to(device)
    class_weights = None
    losses, probs, labels_list = [], [], []
    for i, (records_and_labels) in tqdm(enumerate(dataset), total=len(dataset)):
        records = torch.from_numpy(records_and_labels[0]).float().to(device)
        labels = torch.from_numpy(records_and_labels[1]).float().to(device)
        records = records.transpose(1, 2)
        optimizer.optimizer.zero_grad()
        out = model(records)
        if class_weights is None:
            loss = F.binary_cross_entropy_with_logits(out, labels)
        else:
            loss = F.binary_cross_entropy_with_logits(out, labels, weight=class_weights)
        loss.backward()
        optimizer.step()

        prob = out.sigmoid().data.cpu().numpy()
        losses.append(loss.item())
        probs.append(prob)
        labels_list.append(labels.data.cpu().numpy())

    loss = np.mean(losses)
    labels_list = np.concatenate(labels_list)
    probs = np.concatenate(probs)
    f1_score = sklearn.metrics.f1_score(
        np.argmax(labels_list, axis=1), 
        np.argmax(probs, axis=1), 
        average='macro')
    return loss, f1_score

def main(config):
    train_dataset, val_dataset, _ = get_datasets(config, [i for i in range(12)])
    model = CTN(
        d_model=256, nhead=8, d_ff=2048, 
        num_layers=8, dropout_rate=0.2, deepfeat_sz=64, 
        classes=3).to(device)
    # model = TestArch().to(device)
    
    optimizer = NoamOpt(256, 1, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(30):
        target_labels = [config['arrhythmia_label'], 'NSR', 'Other']
        train_loss, train_f1_macro = train(epoch, model, train_dataset, optimizer, target_labels)
        val_loss, val_f1_macro = validate(epoch, model, val_dataset, optimizer, target_labels)
        print(f'Train - loss: {train_loss}, f1_macro: {train_f1_macro}')
        print(f'Val - loss: {val_loss}, f1_macro: {val_f1_macro}')

    pass
    
if __name__ == "__main__":
    config = read_config(f'{DIR_PATH}/configs/transformer_ext_feats.json')
    main(config)