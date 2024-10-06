import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import tensorflow
from dataset import Dataset, split_dataset_custom, split_dataset
from datahelpers import dldata
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd

import tensorflow.keras as keras
import torch.nn as nn
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
from utils.functions import get_dataset_df
import sklearn.metrics
from sklearn.model_selection import train_test_split
from utils.functions import identify_optimal_thresholds
import tensorflow.keras.callbacks
import shap

tensorflow.compat.v1.disable_v2_behavior()

def get_shap_vals(model, x_train, y_train, x_test, y_test, nsamples=300):
    shap_dp = shap.DeepExplainer(model, x_train[:100])
    shap_vals = shap_dp.shap_values(x_test[:2])
    return shap_vals, np.argmax(y_test, axis=1)

target_labels = ['STD', 'NSR', 'Other']

def get_cl_report(target, predictions):
    predictions = np.argmax(predictions, axis=1)
    cl_report = sklearn.metrics.classification_report(np.argmax(target, axis=1), predictions, target_names=target_labels)
    return cl_report

def get_model(n_classes, length, input_classes, last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(length, input_classes), dtype=np.float32, name='signal')
    x = signal
    num_filters = 128
    for i in range(8):
        x = Conv1D(num_filters, kernel_size, padding='same', use_bias=False,
                kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if i % 2 == 1:
            num_filters /= 2
        if i < 4:
            x = MaxPooling1D(4)(x)

    x = Flatten()(x)
    diagn = Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(signal, diagn)
    return model

def get_dataset(dataframe, config, sequence_len):
    dataset = dldata.DLDataset(dataframe, config['dataset_path'], record_len=sequence_len)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

def main(
    leads,
    fraction_of_train,
    balance_train_samples,
    num_of_batches_test_val,
    num_of_batches_train,
    record_len
    ):
    run_configuration = {
        "leads": leads,
        "fraction_of_train": fraction_of_train,
        "balance_train_samples": balance_train_samples,
        "num_of_batches_test_val": num_of_batches_test_val,
        "num_of_batches_train": num_of_batches_train,
        "record_len": record_len
    }
    dataset_path = "D:\\research_old\\research_large_files\\card_challenge\\training"

    config = {
        "arrhythmia_label": "STD",
        "dataset_path": dataset_path,
        "description_file_name": "dl_train_description.json",
        "val_description_file_name": "dl_val_description.json",
        "test_description_file_name": "test_description.json"
    }

    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])
    test_df = get_dataset_df(config, config['arrhythmia_label'], config['test_description_file_name'])

    train_dataset = get_dataset(train_df, config, record_len)
    val_dataset = get_dataset(val_df, config, record_len)
    test_dataset = get_dataset(test_df, config, record_len)

    all_datasets = [train_dataset, val_dataset, test_dataset]
    for single_dataset in all_datasets:
        single_dataset.set_leads(leads)
        single_dataset.shuffle_df()

        single_dataset.set_limit_batches(num_of_batches_test_val)


    train_dataset.set_limit_batches(num_of_batches_train)
    if balance_train_samples:
        train_dataset.balance_by_max_batch_size()
        train_dataset.shuffle_df()

    model = get_model(3, record_len, len(leads))
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, verbose=1, 
        mode='min', restore_best_weights=True)

    class_weight = train_dataset.get_class_weights(target_labels)

    model.fit(train_dataset, epochs=10, validation_data=val_dataset, 
            verbose=1, class_weight=class_weight, callbacks=[early_stopping])
    # model.load_weights(checkpoint_filepath)

    train_Y = train_dataset.get_all_ys()
    val_Y = val_dataset.get_all_ys()
    test_Y = test_dataset.get_all_ys()
    train_preds = model.predict(train_dataset)
    val_preds = model.predict(val_dataset)
    test_preds = model.predict(test_dataset)
    # train_optimal_thresholds, _ = identify_optimal_thresholds(train_dataset.get_all_ys(), train_preds)
    # val_optimal_thresholds, _ = identify_optimal_thresholds(val_dataset.get_all_ys(), val_preds)

    # train_report = get_cl_report(train_Y, train_preds)
    # val_report = get_cl_report(val_Y, val_preds)
    # test_report = get_cl_report(test_Y, test_preds)

    run_output = {
        "run_configuration": run_configuration,
        "f1_score_macro": {
            "train": sklearn.metrics.f1_score(np.argmax(train_Y, axis=1), np.argmax(train_preds, axis=1), average='macro'),
            "val": sklearn.metrics.f1_score(np.argmax(val_Y, axis=1), np.argmax(val_preds, axis=1), average='macro'),
            "test": sklearn.metrics.f1_score(np.argmax(test_Y, axis=1), np.argmax(test_preds, axis=1), average='macro')
        },
        "f1_score_weighted": {
            "train": sklearn.metrics.f1_score(np.argmax(train_Y, axis=1), np.argmax(train_preds, axis=1), average='weighted'),
            "val": sklearn.metrics.f1_score(np.argmax(val_Y, axis=1), np.argmax(val_preds, axis=1), average='weighted'),
            "test": sklearn.metrics.f1_score(np.argmax(test_Y, axis=1), np.argmax(test_preds, axis=1), average='weighted')
        },
        "auc_macro": {
            "train": sklearn.metrics.roc_auc_score(train_Y, train_preds, average='macro', multi_class='ovo'),
            "val": sklearn.metrics.roc_auc_score(val_Y, val_preds, average='macro', multi_class='ovo'),
            "test": sklearn.metrics.roc_auc_score(test_Y, test_preds, average='macro', multi_class='ovo')
        },
        "auc_weighted": {
            "train": sklearn.metrics.roc_auc_score(train_Y, train_preds, average='weighted', multi_class='ovo'),
            "val": sklearn.metrics.roc_auc_score(val_Y, val_preds, average='weighted', multi_class='ovo'),
            "test": sklearn.metrics.roc_auc_score(test_Y, test_preds, average='weighted', multi_class='ovo')
        },
        "aurpc_macro": {
            "train": sklearn.metrics.average_precision_score(train_Y, train_preds, average='macro'),
            "val": sklearn.metrics.average_precision_score(val_Y, val_preds, average='macro'),
            "test": sklearn.metrics.average_precision_score(test_Y, test_preds, average='macro')
        },
    }
    return run_output

def update_dictionary(dictionary, update):
    dictionary = dictionary.copy()
    for key, val in update.items():
        dictionary[key] = val
    return dictionary

if __name__ == "__main__":
    test_configs = []
    basic_config = {
        "leads": [0,1,2,3,4,5,6,7,8,9,10,11],
        "fraction_of_train": 0.2,
        "balance_train_samples": True,
        "num_of_batches_test_val": 300,
        "num_of_batches_train": 600,
        "record_len": 8000
    }
    test_configs.append(update_dictionary(basic_config, {'leads': [0, 1]}))
    run_outputs = []
    for config in test_configs:
        run_outputs.append(main(config['leads'], config['fraction_of_train'], config['balance_train_samples'], 
             config['num_of_batches_test_val'], config['num_of_batches_train'], 
             config['record_len']))
    run_outputs = {"run_outputs": run_outputs}
    import json
    with open('deeplearning/run_outputs_2.json', 'w') as f:
        json.dump(run_outputs, f, indent=2)