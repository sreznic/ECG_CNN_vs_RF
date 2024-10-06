from models import get_extr_feats_model_multiple_feats_extr
from keras_flops import get_flops as get_keras_flops
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
from calflops import calculate_flops as get_torch_flops
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from keras.models import Model

def get_torch_model(n_classes, leads_num):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    l = []
    in_channels = leads_num
    num_filters = 128
    for i in range(8):
        l.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding='same', bias=False))
        in_channels = num_filters
        l.append(nn.BatchNorm1d(num_filters))
        l.append(nn.ReLU())
        if i % 2 == 1:
            num_filters /= 2
            num_filters = int(num_filters)
        if i < 4:
            l.append(nn.MaxPool1d(4))
    
    l.append(nn.Flatten())
    l.append(nn.Linear(432, 64))
    l.append(nn.ReLU())
    l.append(nn.Linear(64, 32))
    l.append(nn.ReLU())
    l.append(nn.Linear(32, 16))
    l.append(nn.ReLU())
    l.append(nn.Linear(16, 8))
    l.append(nn.ReLU())
    l.append(nn.Linear(8, 4))
    l.append(nn.ReLU())
    l.append(nn.Linear(4, n_classes))
    l.append(nn.Softmax())
    model = nn.Sequential(*l)
    return model

def get_extr_feats_model_multiple_feats_extr(n_classes, length, leads_num=1, last_layer='softmax'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(length, leads_num), dtype=np.float32, name='signal')
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
    x = Dense(64, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(32, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(16, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(8, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(4, activation='relu', kernel_initializer=kernel_initializer)(x)
    diagn = Dense(n_classes, activation=last_layer)(x)
    model = Model(signal, diagn)
    return model

def main():
    torch_flops_leads = []
    for lead_num in [i for i in range(1, 13)]:
        model = get_extr_feats_model_multiple_feats_extr(3, 7000, lead_num)
        # torch_flp = get_torch_flops(torch_model, (1, lead_num, 7000), output_precision=15)
        torch_flp = get_keras_flops(model, 1)
        torch_flops_leads.append(torch_flp)
    top = torch_flops_leads[-1]
    for i, tp_leads in enumerate(torch_flops_leads):
        print(i, tp_leads, f"{tp_leads * 100 / top:.2f}")
    pass

if __name__ == "__main__":
    main()