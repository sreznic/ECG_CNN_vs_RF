# Add the directory of the parent folder to the system path
import sys
sys.path.append('..')

from mlpython.utils.functions import read_config, read_description
import os
import scipy.io as sio
import pandas as pd
import numpy as np
import neurokit2 as nk
import scipy.stats
from tqdm import tqdm


# Extract features from the header and recording.
def get_features(header, recording, preprocess_configs=None):
    features = np.zeros(12, dtype=np.object_)
    age_sex = get_age_sex(header)
    arrhythmias = get_arrhythmias(header)
    rr_mean_std, is_error = get_rr_mean_std(recording, preprocess_configs=preprocess_configs)

    features[0] = arrhythmias
    features[1:4] = age_sex
    features[4:] = rr_mean_std
    return features, is_error

def get_age_sex(header):
    """
    Return output = np.array([age, man, woman])
    age_default is set to 0.6.
    """
    output = np.zeros(3, dtype=np.float32)

    # Make age and age_mask.
    age_default = 0.6
    age = get_age(header)
    if age is None or np.isnan(age):
        age = age_default
    else:
        age = age / 100

    # Make man, woman, and sex_mask.
    sex = get_sex(header)
    if sex in ("Female", "female", "F", "f"):
        man = 0
        woman = 1
        sex_mask = 0
    elif sex in ("Male", "male", "M", "m"):
        man = 1
        woman = 0
        sex_mask = 0
    else:
        man = 0
        woman = 0
        sex_mask = 1

    output[0] = age
    output[1] = man
    output[2] = woman
    return output

def get_arrhythmias(header):
    for l in header.split('\n'):
        if l.startswith('# Dx'):
            arrhythmias = l[6:].strip()
            return arrhythmias

# Get age from header.
def get_age(header):
    age = None
    for l in header.split('\n'):
        if l.startswith('# Age'):
            try:
                age = float(l.split(': ')[1].strip())
            except:
                age = float('nan')
    return age

# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('# Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

def get_rr_mean_std(recording, preprocess_configs=None):
    """
    Extract mean, std, RMSSD of RR interval length and mean, RMSSD of R peak value, HR mean, HR min, and HR max from lead 2. 
    If lead 2 does not work, try with lead 1.
    If lead 1 & lead 2 do not work, put default value.
    Output : (16,)-ndarry. 
             output = np.array([(mean of RR interval length), (mask-mean of RR interval length), (std of RR interval length),
                                (mask-std of RR interval length), (RMSSD of RR interval length), (mask-RMSSD of RR interval length),
                                (mean of R peak value), (mask-mean of R peak value), (RMSSD of R peak value), (mask-RMSSD of R peak value),
                                (HR_mean), (mask-HR_mean), (HR_min), (mask-HR_min), (HR_max), (mask-HR_max)])
    """
    resample_freq = preprocess_configs["resample_freq"]
    output = np.zeros(16, dtype=np.float32)
    default_value = -1
    try_lead_1 = False
    is_error = False

    mask_mean_RR_interval = 0
    mask_std_RR_interval = 0
    mask_RMSSD_RR_interval = 0
    mask_mean_RR_peaks = 0
    mask_RMSSD_RR_peaks = 0
    mask_HR_mean = 0
    mask_HR_min = 0
    mask_HR_max = 0

    # Try with lead 2 (lead idx 1)
    try:
        filtered_data0 = recording[1].copy()
        _, R_peaks = nk.ecg_peaks(filtered_data0, sampling_rate=resample_freq)   # This may occur an error. Thus, I used [try & except]
        R_peaks = R_peaks['ECG_R_Peaks']
        if R_peaks.shape[0] < 4:
            try_lead_1 = True
        else: 
            diff_RR = (R_peaks[1:] - R_peaks[:-1]) / resample_freq                   # freq로 나눔으로서, 단위가 second가 됨. This may occur an error.
            mean_RR_interval = diff_RR.mean()
            std_RR_interval = diff_RR.std()
            RMSSD_RR_interval = get_RMSSD(diff_RR.astype(np.float64))
            mean_RR_peaks = filtered_data0[R_peaks].mean()
            RMSSD_RR_peaks = get_RMSSD(filtered_data0[R_peaks].astype(np.float64))
            rate = nk.signal.signal_rate(R_peaks, sampling_rate=resample_freq, desired_length=len(filtered_data0))
            HR_mean = rate.mean() / 100
            HR_min = rate.min() / 100
            HR_max = rate.max() / 100
            if (diff_RR < 0.2).sum() != 0 or mean_RR_interval > 2.5 or std_RR_interval > 0.215 :
                try_lead_1 = True
    except:
        try_lead_1 = True
    
    # Try with lead 1 (lead idx 0)
    if try_lead_1:
        try:
            filtered_data0 = (recording[0].copy())
            _, R_peaks = nk.ecg_peaks(filtered_data0, sampling_rate=resample_freq)   # This may occur an error. Thus, I used [try & except]
            R_peaks = R_peaks['ECG_R_Peaks']
            if R_peaks.shape[0] < 4:
                is_error = True
            else: 
                diff_RR = (R_peaks[1:] - R_peaks[:-1]) / resample_freq                   # freq로 나눔으로서, 단위가 second가 됨. This may occur an error.
                mean_RR_interval = diff_RR.mean()
                std_RR_interval = diff_RR.std()
                RMSSD_RR_interval = get_RMSSD(diff_RR.astype(np.float64))
                mean_RR_peaks = filtered_data0[R_peaks].mean()
                RMSSD_RR_peaks = get_RMSSD(filtered_data0[R_peaks].astype(np.float64))
                rate = nk.signal.signal_rate(R_peaks, sampling_rate=resample_freq, desired_length=len(filtered_data0))
                HR_mean = rate.mean() / 100
                HR_min = rate.min() / 100
                HR_max = rate.max() / 100
                # if (diff_RR < 0.2).sum() != 0 or mean_RR_interval > 2.5 or std_RR_interval > 0.215:
                #     is_error = True
        except:
            is_error = True
    
    # Cannot extract exact R pick locations. Put default values.
    if is_error:
        mean_RR_interval = default_value
        std_RR_interval = default_value
        RMSSD_RR_interval = default_value
        mean_RR_peaks = default_value
        RMSSD_RR_peaks = default_value
        HR_mean = default_value
        HR_min = default_value
        HR_max = default_value
        mask_mean_RR_interval = 1
        mask_std_RR_interval = 1
        mask_RMSSD_RR_interval = 1
        mask_mean_RR_peaks = 1
        mask_RMSSD_RR_peaks = 1
        mask_HR_mean = 1
        mask_HR_min = 1
        mask_HR_max = 1

    output[0] = mean_RR_interval
    output[1] = mask_mean_RR_interval
    output[2] = std_RR_interval
    output[3] = mask_std_RR_interval
    output[4] = RMSSD_RR_interval   
    output[5] = mask_RMSSD_RR_interval
    output[6] = mean_RR_peaks
    output[7] = mask_mean_RR_peaks
    output[8] = RMSSD_RR_peaks
    output[9] = mask_RMSSD_RR_peaks
    output[10] = HR_mean
    output[11] = mask_HR_mean
    output[12] = HR_min
    output[13] = mask_HR_min
    output[14] = HR_max
    output[15] = mask_HR_max
    return output[::2], is_error

def get_RMSSD(intervals):
    return (sum(np.power(intervals, 2) ** 2) / (len(intervals) - 1)) ** 0.5

def main():
    config = read_config('configs/extract_adv_feats_config.json')

    description_path = os.path.join(config['dataset_path'], config['description_file_name'])
    description = read_description(description_path)

    record_names_set = set()
    features = []
    errors_num = 0

    for dataset in tqdm(description.keys()):
        for arrhythmia in tqdm(description[dataset].keys(), leave=False):
            for record in tqdm(description[dataset][arrhythmia], leave=False):
                if record in record_names_set:
                    continue
                hea_path = os.path.join(config['dataset_path'], record + '.hea')
                mat_path = os.path.join(config['dataset_path'], record + '.mat')
                if not os.path.exists(hea_path) or not os.path.exists(mat_path):
                    continue
                mat = sio.loadmat(mat_path)['val']
                with open(hea_path, 'r') as f:
                    header = f.read()
                conf = {}
                conf["resample_freq"] = 300
                conf["use_filter"] = True
                conf["use_standardization"] = True
                conf["use_max_min_normalization"] = False

                feats, is_error = get_features(header, mat, conf)
                errors_num += 1 if is_error else 0

                features.append([dataset, record] + feats.tolist())

    column_names = ["dataset", "record", "arrhythmias"] + \
        [f"feat{i}" for i in range(1, 12)]
    
    df = pd.DataFrame(features, columns=column_names)
    df.to_csv(config['extract_feats_path'], index=False)
    print(f"Number of errors: {errors_num}")

if __name__ == "__main__":
    main()