from dataset import Dataset
from datahelpers import dldata
from datahelpers.dldata import DLDatasetDataAugm
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd

import torch.nn as nn
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from keras.models import Model
from utils.functions import get_dataset_dataframe
import sklearn.metrics
from sklearn.model_selection import train_test_split
from utils.functions import identify_optimal_thresholds

ARRHYTHMIA_LABELS = ['PVC', 'NSR', 'Other']
arrhythmia_label = ARRHYTHMIA_LABELS[0]

config = {
    "arrhythmia_label": arrhythmia_label,
    "dataset_path": "D:\\research_old\\research_large_files\\card_challenge\\training",
    "description_file_name": "train_description.json",
    "features_path": "features/all/whole_ds_processed.csv",
    "ds_class_module": "datahelpers.allpreddata",
    "ds_class_name": "AllPredData",
    "test_description_file_name": "test_description.json"
}

target_labels = ARRHYTHMIA_LABELS

def get_dataset(dataframe):
    dataset = dldata.DLDataset(dataframe, config['dataset_path'], record_len=5000)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

def save_dataset(df, path, batches_num, augment):
    dataset = get_dataset(df)
    dataset.set_limit_batches(batches_num)
    dataset.balance_by_max_batch_size()

    augm_dataset = DLDatasetDataAugm()
    if augment:
        augm_dataset.augment(dataset)
    else:
        augm_dataset.noaugment(dataset)
    augm_dataset.save_records(path)

train_val_df = get_dataset_dataframe(config, config['arrhythmia_label'], False)
train_df, val_df = train_test_split(train_val_df, test_size=0.15)
test_df = get_dataset_dataframe(config, config['arrhythmia_label'], True)

save_dataset(
    train_df, 
    f'augm_data/train_{arrhythmia_label}',
    700,
    True)

save_dataset(
    val_df, 
    f'augm_data/val_{arrhythmia_label}',
    700,
    False)

save_dataset(
    test_df, 
    f'augm_data/test_{arrhythmia_label}',
    700,
    False)