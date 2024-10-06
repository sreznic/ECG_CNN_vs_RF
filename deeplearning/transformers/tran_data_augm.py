import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from dataset import Dataset
import scipy.io as sio
from utils.functions import resample_to_500hz

DATASET_PATH = "D:\\research_old\\research_large_files\\card_challenge\\training"
TRAIN_DESCRIPTION_FILENAME = "dl_train_description.json"
VAL_DESCRIPTION_FILENAME = "dl_val_description.json"
TEST_DESCRIPTION_FILENAME = "test_description.json"
WHOLE_DESCRIPTION_FILENAME = "dataset_description.json"

def get_augm_noise(noise):
    def fun(signals):
        signals = signals.copy()
        for i in range(signals.shape[0]):
            signals[i] = signals[i] + noise * np.random.randn(signals[i].shape[0])
        return signals
    return fun

def get_time_warping(factor):
    def fun(signals):
        signals = signals.copy()
        for i in range(signals.shape[0]):
            new_time = np.linspace(0, 1, signals.shape[1])
            warped_time = new_time / factor
            interp = interp1d(warped_time, signals[i], kind='cubic', fill_value="extrapolate")
            signals[i] = interp(new_time)
        return signals
    return fun

def get_amplitude_scaling(factor):
    def fun(signals):
        signals = signals.copy()
        return signals * factor
    return fun

def get_shifted(shift_size):
    def fun(signals):
        signals = signals.copy()
        for i in range(signals.shape[0]):
            signals[i] = np.roll(signals[i], shift_size)
        return signals
    return fun

def combine_funs(funs):
    def fun(signals, spikes):
        for f in funs:
            signals = f(signals)
        return signals
    return fun

def get_augmented(original_data, apply_augmentations, arrhythmia_label):
    signals = original_data
    augmentation_functions = [
        get_augm_noise(0.1), get_augm_noise(0.2), 
        get_time_warping(0.5), get_time_warping(0.8), get_time_warping(1.2), get_time_warping(1.5),
        get_amplitude_scaling(0.5), get_amplitude_scaling(2),
        get_shifted(1000), get_shifted(-1000), get_shifted(4000), get_shifted(8000), # 11
        combine_funs([get_augm_noise(0.1), get_time_warping(0.5)]),
        combine_funs([get_augm_noise(0.2), get_amplitude_scaling(0.5)]),
        combine_funs([get_augm_noise(0.1), get_amplitude_scaling(2)]),
        combine_funs([get_augm_noise(0.2), get_shifted(1000)]),
        combine_funs([get_augm_noise(0.2), get_shifted(-1000)]),
        combine_funs([get_time_warping(0.5), get_amplitude_scaling(0.5)]),
        combine_funs([get_time_warping(0.5), get_amplitude_scaling(2)]),
        combine_funs([get_time_warping(0.5), get_shifted(1000)]),
        ]
    augmented_signals = np.zeros(
        (signals.shape[0] * apply_augmentations, signals.shape[1], signals.shape[2]))
    i = 0
    for augm in tqdm(augmentation_functions[:apply_augmentations], desc=f'Augmenting {arrhythmia_label}'):
        for j in range(signals.shape[0]):
            augmented_signals[i] = augm(signals[j])
            i += 1

    total_signals = np.concatenate((signals, augmented_signals), 0)

    signals = torch.from_numpy(total_signals)
    return signals

def get_records(dataset_path, df, arrhythmia_label, record_len):
    records = np.zeros((len(df), 12, record_len))
    for i in tqdm(range(len(df)), desc=f'Getting data {arrhythmia_label}'):
        record_name = df.iloc[i]['Record']
        record_path = os.path.join(dataset_path, record_name)
        record = sio.loadmat(record_path)['val']
        # Normalize in range -1, 1

        record = (record - np.mean(record, axis=1)[..., np.newaxis]) / (np.std(record, axis=1)[..., np.newaxis] + 1e-10)
        with open(record_path + '.hea', 'r') as f:
            lines = f.readlines()
            frequency = int(lines[0].split(' ')[2])
        record = resample_to_500hz(record, frequency)
        if record.shape[1] < record_len:
            left_pad = (record_len - record.shape[1]) // 2
            right_pad = record_len - record.shape[1] - left_pad
            record = np.pad(record, ((0, 0), (left_pad, right_pad)))
        elif record.shape[1] > record_len:
            # Take middle
            start = (record.shape[1] - record_len) // 2
            record = record[:, start:start+record_len]
        records[i] = record
    return records

def get_datasets(
        dataset_path, description_file_name, arrhythmia_label,\
        sample_records, max_transformations, record_len):
    ds = Dataset(dataset_path, description_file_name)
    df = ds.get_pandas_dataframe(arrhythmia_label)
    arrhythm_df = df[df['Arrhythmia'] == arrhythmia_label]
    nsr_df = df[df['Arrhythmia'] == 'NSR']
    nsr_df = nsr_df[nsr_df['IsMultipleArrhythmias'] == False]
    other_df = df[df['Arrhythmia'] == 'Other']
    other_df = other_df[other_df['IsMultipleArrhythmias'] == False]
    arrhythm_df = arrhythm_df.sample(min(sample_records, len(arrhythm_df))).reset_index(drop=True)
    nsr_df = nsr_df.sample(min(sample_records, len(nsr_df))).reset_index(drop=True)
    other_df = other_df.sample(min(sample_records, len(other_df))).reset_index(drop=True)

    arrhythm_needed_transformations = min(sample_records // (len(arrhythm_df) + 1), max_transformations)
    nsr_needed_transformations = min(sample_records // (len(nsr_df) + 1), max_transformations)
    other_needed_transformations = min(sample_records // (len(other_df) + 1), max_transformations)
    arrhythm_records = get_records(dataset_path, arrhythm_df, arrhythmia_label, record_len)
    nsr_records = get_records(dataset_path, nsr_df, 'NSR', record_len)
    other_records = get_records(dataset_path, other_df, 'Other', record_len)
    arrhythm_records = get_augmented(arrhythm_records, arrhythm_needed_transformations, arrhythmia_label)
    nsr_records = get_augmented(nsr_records, nsr_needed_transformations, 'NSR')
    other_records = get_augmented(other_records, other_needed_transformations, 'Other')

    arrhythm_records = arrhythm_records[:sample_records]
    nsr_records = nsr_records[:sample_records]
    other_records = other_records[:sample_records]
    return {'arrhythmia': arrhythm_records, 'NSR': nsr_records, 'Other': other_records}

def no_augm_get_datasets(
        dataset_path, description_file_name, arrhythmia_label,\
        sample_records, record_len):
    ds = Dataset(dataset_path, description_file_name)
    df = ds.get_pandas_dataframe(arrhythmia_label)

    arrhythm_df = df[df['Arrhythmia'] == arrhythmia_label]
    nsr_df = df[df['Arrhythmia'] == 'NSR']
    nsr_df = nsr_df[nsr_df['IsMultipleArrhythmias'] == False]
    other_df = df[df['Arrhythmia'] == 'Other']
    other_df = other_df[other_df['IsMultipleArrhythmias'] == False]

    total_len = len(arrhythm_df) + len(nsr_df) + len(other_df)
    arrhythm_df_perc = len(arrhythm_df) / total_len
    nsr_df_perc = len(nsr_df) / total_len
    other_df_perc = len(other_df) / total_len

    sample_records = min(sample_records, total_len)
    arrhythm_count = min(len(arrhythm_df), int(sample_records * arrhythm_df_perc))
    nsr_count = min(len(nsr_df), int(sample_records * nsr_df_perc))
    other_count = min(len(other_df), int(sample_records * other_df_perc))

    arrhythm_df = arrhythm_df.sample(arrhythm_count).reset_index(drop=True)
    nsr_df = nsr_df.sample(nsr_count).reset_index(drop=True)
    other_df = other_df.sample(other_count).reset_index(drop=True)

    arrhythm_records = get_records(dataset_path, arrhythm_df, arrhythmia_label, record_len)
    nsr_records = get_records(dataset_path, nsr_df, 'NSR', record_len)
    other_records = get_records(dataset_path, other_df, 'Other', record_len)

    arrhythm_records = torch.from_numpy(arrhythm_records)
    nsr_records = torch.from_numpy(nsr_records)
    other_records = torch.from_numpy(other_records)

    return {'arrhythmia': arrhythm_records, 'NSR': nsr_records, 'Other': other_records}

def main():
    ds = get_datasets('STE', 500, 10, 10000)
    pass

if __name__ == "__main__":
    main()