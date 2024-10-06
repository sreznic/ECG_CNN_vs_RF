import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
import argparse
from keras.models import Model
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
from utils.functions import read_config, get_dataset_df, get_all_records, save_log_as_npy
import datahelpers.dldata as dldata
import math
from keras.utils import Sequence
from models import get_extr_feats_model_multiple_feats_extr, model_with_needed_layers, get_feats_model
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
import tensorflow as tf
from tqdm import tqdm
import random
from time import time
import json
from utils.constants import LEAD_NAMES
from models import get_feats_model
import shap
import sklearn.metrics
from dl_optimal_subsets import DatasetDictionaryManager

def get_dataset(dataframe, config, batches, 
                sequence_len, target_labels, features_dict):
    dataset = dldata.DLDataset(
        dataframe, config['dataset_path'], record_len=sequence_len)
    dataset.set_record_lookup(features_dict)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    dataset.set_size_version_2(True)
    dataset.set_limit_batches(batches)
    dataset.shuffle_df()

    return dataset

for _ in range(10):
    print("REPEAT")
    DIR_PATH = "deeplearning/with_feats_extr"
    ARRHYTHMIA = "IAVB"

    config = read_config(f'{DIR_PATH}/configs/dl_optimal_subsets.json')
    config["arrhythmia_label"] = ARRHYTHMIA

    features = np.load(f'{config["features_path"]}/{ARRHYTHMIA}.npy', allow_pickle=True).item()

    target_labels = [ARRHYTHMIA, 'NSR', 'Other']

    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])
    test_df = get_dataset_df(config, config['arrhythmia_label'], config['test_description_file_name'])

    train_dataset = get_dataset(train_df, config, config["batches"]["train"], config['feats_num'], target_labels, features)
    val_dataset = get_dataset(val_df, config, config["batches"]["val"], config['feats_num'], target_labels, features)
    test_dataset = get_dataset(test_df, config, config["batches"]["test"], config['feats_num'], target_labels, features)


    lead_num = 5
    model = get_feats_model(
        32, lead_num, 5, False, 0.1
    )
    optimizer = Adam()
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=4, verbose=1, 
        mode='max', restore_best_weights=True)
    ds_manager = DatasetDictionaryManager(train_dataset, val_dataset, test_dataset, target_labels)
    dataset_dictionary = ds_manager.get_dataset_dictionary([0,5,7,3,4])
    model.fit(x=dataset_dictionary['train']['xs'], 
            y=dataset_dictionary['train']['ys'], 
            epochs=2, 
            validation_data=(dataset_dictionary['val']['xs'], dataset_dictionary['val']['ys']),
            callbacks=[early_stopping])#,

    print("Train classification report:")
    print(sklearn.metrics.classification_report(
        np.argmax(dataset_dictionary['train']['ys'], axis=1), 
        np.argmax(model.predict(dataset_dictionary['train']['xs']), axis=1), 
        target_names=target_labels))
    print("Val classification report:")
    print(sklearn.metrics.classification_report(
        np.argmax(dataset_dictionary['val']['ys'], axis=1),
        np.argmax(model.predict(dataset_dictionary['val']['xs']), axis=1),
        target_names=target_labels))
    print("Test classification report:")
    print(sklearn.metrics.classification_report(
        np.argmax(dataset_dictionary['test']['ys'], axis=1),
        np.argmax(model.predict(dataset_dictionary['test']['xs']), axis=1),
        target_names=target_labels))

    pass