import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm as tqdm_fun
import random

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
    def fun(signals):
        for f in funs:
            signals = f(signals)
        return signals
    return fun

def get_augmented_data(data, num_funs, tqdm=False):
    new_data = np.zeros((data.shape[0] * (num_funs + 1), data.shape[1], data.shape[2]))
    new_data[:data.shape[0]] = data
    augmented_functions = [
        get_augm_noise(0.05), get_augm_noise(0.1), get_augm_noise(0.2),
        get_time_warping(0.2), get_time_warping(0.5), get_time_warping(0.7), get_time_warping(0.9),
        get_amplitude_scaling(0.5), get_amplitude_scaling(2),
        get_shifted(500), get_shifted(-1000), get_shifted(1500), get_shifted(3000),
        combine_funs([get_augm_noise(0.1), get_time_warping(0.2)]),
        combine_funs([get_augm_noise(0.1), get_time_warping(0.5)]),
        combine_funs([get_augm_noise(0.1), get_time_warping(0.7)]),
        combine_funs([get_augm_noise(0.1), get_time_warping(0.9)]),
        combine_funs([get_augm_noise(0.2), get_amplitude_scaling(0.5)]),
        combine_funs([get_augm_noise(0.1), get_amplitude_scaling(2)]),
        combine_funs([get_augm_noise(0.2), get_shifted(1000)]),
        combine_funs([get_augm_noise(0.2), get_shifted(-1000)]),
        combine_funs([get_time_warping(0.5), get_amplitude_scaling(0.5)]),
        combine_funs([get_time_warping(0.5), get_amplitude_scaling(2)]),
        combine_funs([get_time_warping(0.5), get_shifted(1000)]),
    ]
    random.shuffle(augmented_functions)
    num_funs = min(num_funs, len(augmented_functions))
    iterr = tqdm_fun(range(num_funs)) if tqdm else range(num_funs)
    for i in iterr:
        augm = augmented_functions[i]
        s = (i + 1) * data.shape[0]
        for j in range(data.shape[0]):
            new_data[s + j] = augm(data[j])
    return new_data

if __name__ == "__main__":
    # Generate 10 12-lead ECG with 5000 samples
    data = np.random.randn(10, 12, 5000)
    t = get_augmented_data(data, 10)
    pass