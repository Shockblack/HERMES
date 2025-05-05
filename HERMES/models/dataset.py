import os
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

# Create a custom dataset class that inherits from DatasetFolder

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spectra, parameters, rtop = sample['spectra'], sample['parameters'], sample['rtop']

        return {'spectra': torch.from_numpy(spectra),
                'parameters': torch.from_numpy(parameters),
                'rtop': torch.from_numpy(rtop)}

class Normalize(object):

    def __call__(self, sample, normalization_factor=1e-3):
        spectra, parameters, rtop = sample['spectra'], sample['parameters'], sample['rtop']

        spectra = spectra / normalization_factor

        return {'spectra': spectra,
                'parameters': parameters,
                'rtop': rtop}

class spectraDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.parameters = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spec_name = os.path.join(self.root_dir, self.parameters.iloc[idx, 0])
        
        spectra = np.loadtxt(spec_name, dtype=np.float32)
        inputs = np.array(self.parameters.iloc[idx, 1:-1], dtype=np.float32)
        rtop = np.array(self.parameters.iloc[idx, -1], dtype=np.float32)

        sample = {'spectra': spectra, 'parameters': inputs, 'rtop': rtop}

        if self.transform:
            sample = self.transform(sample)

        return sample
