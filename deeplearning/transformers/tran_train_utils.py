import datahelpers.dldata as dldata
import numpy as np
from utils.functions import read_config, get_dataset_df, get_all_records
from tran_data_augm import get_datasets as augm_get_datasets, no_augm_get_datasets
from tran_dataset import TorchECGDatasetDataDict, TorchECGDatasetKeras
import os
from torch.utils.data import DataLoader

def get_datasets(config, arrhythmia_label, leads, debug):
    train_df = get_dataset_df(config, arrhythmia_label, config['train_description_file_name'])
    val_df = get_dataset_df(config, arrhythmia_label, config['val_description_file_name'])
    test_df = get_dataset_df(config, arrhythmia_label, config['test_description_file_name'])

    target_labels = [arrhythmia_label, 'NSR', 'Other']
    
    train_dataset = get_dataset(train_df, config, config['record_len'], target_labels)
    val_dataset = get_dataset(val_df, config, config['record_len'], target_labels)
    test_dataset = get_dataset(test_df, config, config['record_len'], target_labels)
    if debug:
        train_dataset.set_limit_batches(config['debug_num_batches'])
        train_dataset.balance_by_max_batch_size()
        val_dataset.set_limit_batches(config['debug_num_batches'])
        test_dataset.set_limit_batches(config['debug_num_batches'])
    else:
        train_dataset.set_limit_batches(config['train_batches'])
        train_dataset.balance_by_max_batch_size()
        val_dataset.set_limit_batches(config['val_batches'])
        test_dataset.set_limit_batches(config['test_batches'])

    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.shuffle_df()
        dataset.set_leads(leads)

    train_dataset = TorchECGDatasetKeras(train_dataset)
    val_dataset = TorchECGDatasetKeras(val_dataset)
    test_dataset = TorchECGDatasetKeras(test_dataset)

    return train_dataset, val_dataset, test_dataset

def get_augm_datasets(
        config, arrhythmia_label, 
        sample_records_train_each, sample_records_val_test_total,
        max_transformations, record_len):
    train_data = augm_get_datasets(
        config['dataset_path'], config['train_description_file_name'],\
        arrhythmia_label, sample_records_train_each, max_transformations,\
        record_len)
    val_data = no_augm_get_datasets(
        config['dataset_path'], config['val_description_file_name'],\
        arrhythmia_label, sample_records_val_test_total, record_len)
    test_data = no_augm_get_datasets(
        config['dataset_path'], config['test_description_file_name'],\
        arrhythmia_label, sample_records_val_test_total, record_len)
    
    keys = ['arrhythmia', 'NSR', 'Other']

    train_dataset = TorchECGDatasetDataDict(train_data, keys)
    val_dataset = TorchECGDatasetDataDict(val_data, keys)
    test_dataset = TorchECGDatasetDataDict(test_data, keys)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader, train_dataset.get_weights()

def get_dataset(dataframe, config, sequence_len, target_labels):
    dataset = dldata.DLDataset(dataframe, config['dataset_path'], record_len=sequence_len)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

def start_log(loc, arrhythmia, leads):
    path = os.path.join(loc, f'log_{arrhythmia}{len(leads)}.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('epoch,train_loss,train_f1,val_loss,val_f1\n')

def write_log(loc, arrhythmia, leads, epoch, train_loss, train_f1, val_loss, val_f1):
    path = os.path.join(loc, f'log_{arrhythmia}{len(leads)}.csv')
    with open(path, 'a') as f:
        f.write(f'{epoch},{train_loss},{train_f1},{val_loss},{val_f1}\n')