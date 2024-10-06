import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

from keras.optimizers import Adam
from sklearn.utils import class_weight
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from utils.constants import ARRHYTHMIA_LABELS
from utils.functions import read_config, get_dataset_df, get_all_records
from models import get_extr_feats_model_multiple_feats_extr, optimal_feats_model
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
from keras.models import Model
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import sklearn.metrics
import json
from dl_utils import get_whole_dataset
import itertools
import datahelpers.dldata as dldata
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
from wcce import WeightedCategoricalCrossentropy
from time import time


DIR_PATH = "deeplearning/with_feats_extr"

LOSS_NAME = 'categorical_crossentropy'
def get_dataset(dataframe, config, sequence_len, target_labels):
    dataset = dldata.DLDataset(dataframe, config['dataset_path'], record_len=sequence_len)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

def get_datasets(config):
    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])
    test_df = get_dataset_df(config, config['arrhythmia_label'], config['test_description_file_name'])

    target_labels = [config['arrhythmia_label'], 'NSR', 'Other']
    
    train_dataset = get_dataset(train_df, config, config['record_len'], target_labels)
    val_dataset = get_dataset(val_df, config, config['record_len'], target_labels)
    test_dataset = get_dataset(test_df, config, config['record_len'], target_labels)

    train_dataset.set_limit_batches(config['train_batches'])
    train_dataset.balance_by_max_batch_size()
    val_dataset.set_limit_batches(config['val_batches'])
    test_dataset.set_limit_batches(config['test_batches'])
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.shuffle_df()
        dataset.set_leads(config['leads'])

    return train_dataset, val_dataset, test_dataset

def get_whole_dataset(config):
    records_names = get_all_records(config, config['whole_description_file_name'])
    records_names = records_names
    whole_df = pd.DataFrame({
        'Record': records_names,
        'Arrhythmia': ['Dummy'] * len(records_names)
    })
    dl_dataset = DLDatasetNoBatch(whole_df, config['dataset_path'], config['record_len'])
    dl_dataset.set_transform_y(lambda _: [0, 0, 0])
    dl_dataset.set_leads(config['leads'])
    data = np.array([dl_dataset[i][0] for i in range(len(dl_dataset))])
    return data, records_names

def get_results_arrhhythm_lead(config):
    arrhythmia_label = "AF"
    train_dataset, val_dataset, _ = get_datasets(config)
    for _ in tqdm(train_dataset, desc="Train dataset"):
        pass
    for _ in tqdm(val_dataset, desc="Val dataset"):
        pass
    opt = Adam(config['learning_rate'])
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   mode='min',
                                    factor=0.1,
                                    patience=3,
                                    min_lr=config['learning_rate'] / 100),
                    EarlyStopping(monitor='val_loss',
                                patience=5,  # Patience should be larger than the one in ReduceLROnPlateau
                                min_delta=0.00001,
                                restore_best_weights=True)]
    model = get_extr_feats_model_multiple_feats_extr(3, config['record_len'], leads_num=len(config['leads']))
    model.compile(loss=LOSS_NAME, optimizer=opt, metrics=['acc'])
    
    train_y = train_dataset.get_all_ys()
    lbl_train_y = np.argmax(train_y, axis=1)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(lbl_train_y), y=lbl_train_y)
    class_weights = { i : class_weights[i] for i in range(len(class_weights)) }
    start_time = time()
    history = model.fit(train_dataset,
                        epochs=config['epochs'],
                        validation_data=val_dataset,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=1)
    end_time = time()
    # Get validation F1 score and loss
    val_y = val_dataset.get_all_ys()
    val_pred = model.predict(val_dataset)
    f1_score = sklearn.metrics.f1_score(
        np.argmax(val_y, axis=1), 
        np.argmax(val_pred, axis=1), 
        average='macro')

    weights2 = np.array([class_weights[i] for i in range(len(class_weights))])
    wcce = WeightedCategoricalCrossentropy(weights2)
    loss = wcce(val_y, val_pred).numpy()
    return float(f1_score), float(loss), (end_time - start_time)

def main(config):
    results = {}
    record_lens = [500, 1000, 2500, 5000, 7000, 10000]
    leads = [[0], [i for i in range(12)]]
    epochs = [10, 30]
    combinations = list(itertools.product(record_lens, leads, epochs))
    for iteration in range(9,15):
        for comb in combinations:
            record_length = comb[0]
            leads = comb[1]
            epochs = comb[2]
            config['record_len'] = record_length
            config['leads'] = leads
            config['epochs'] = epochs
            config['arrhythmia_label'] = "AF"
            results[f"{comb}"] = get_results_arrhhythm_lead(config)
        json.dump(results, open(f'{config["save_log_path"]}/optimal_record_len{iteration}.json', 'w'))

if __name__ == "__main__":
    config = read_config(f'{DIR_PATH}/configs/optimal_signal_length.json')
    main(config)