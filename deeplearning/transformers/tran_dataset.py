from torch.utils.data import Dataset
import torch

class TorchECGDatasetKeras(Dataset):
    def __init__(self, kerasds):
        self.kerasds = kerasds
    
    def __len__(self):
        return len(self.kerasds)
    
    def __getitem__(self, idx):
        items = self.kerasds[idx]
        return torch.from_numpy(items[0]), torch.from_numpy(items[1])
    
class TorchECGDatasetDataDict(Dataset):
    def __init__(self, data_dict, keys):
        total_length = sum([len(data_dict[key]) for key in keys])
        ecg_shape = data_dict[keys[0]][0].shape
        self.data = torch.zeros((total_length, ecg_shape[0], ecg_shape[1]))
        self.labels = torch.zeros((total_length, len(keys)))
        i = 0
        for key in keys:
            signals = data_dict[key]
            self.data[i:i+len(signals)] = signals
            self.labels[i:i+len(signals), keys.index(key)] = 1
            i = i + len(signals)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_weights(self):
        num_samples = len(self.labels)
        weights = torch.zeros(self.labels.shape[1])
        for i in range(self.labels.shape[1]):
            class_count = torch.sum(self.labels[:, i] == 1)
            weights[i] = num_samples / (self.labels.shape[1] * class_count)
        weights = weights / torch.sum(weights)
        return weights
