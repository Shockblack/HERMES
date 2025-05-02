import os
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# Create a custom dataset class that inherits from DatasetFolder

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spectra, parameters = sample['spectra'], sample['parameters']

        return {'spectra': torch.from_numpy(image),
                'parameters': torch.from_numpy(landmarks)}

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
        
        spectra = np.loadtxt(spec_name)
        inputs = np.array(self.parameters.iloc[idx, 1:], dtype=np.float32)

        sample = {'spectra': spectra, 'parameters': inputs}

        if self.transform:
            sample = self.transform(sample)

        return sample
