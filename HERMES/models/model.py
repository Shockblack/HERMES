import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from blocks import *

class SpecGen(nn.Module):
    """
    A class that generates spectra from planetary parameters.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the SpecGen model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(SpecGen, self).__init__()
        self.fc1 = nn.Linear(in_channels, 64) # (13 --> 64) --> (bsz, 1, 64)
        self.conv1 = convBlock(1, 63) # Because we torch.cat the input and output of the convBlock, we need to add the input channels to the output channels (bsz, 64, ??)
        self.conv2 = convBlock(64, 64)
        self.conv3 = convBlock(128, 128)
        self.conv4 = convBlock(256, 256)
        self.conv5 = convBlock(512, 512, final_layer=True) # 1024 * 64 = 64194
        # Flattening
        self.fc2 = nn.Linear(1024 * 64, out_channels) # (flatenned size --> out_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the SpecGen model.

        Args:
            x (Tensor): Input tensor.
        """
        x = F.relu(self.fc1(x))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #Flatten
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x

def L_phys_delta(model, inputs, gamma):
    """
    Physical loss term that constrains the transit depth to be between (R_p/R_s)**2 and (R_p+H/R_s)**2
    where H is the height of the atmosphere model. We define R_top = R_p + H.

    Args:
        model (nn.Module): The SpecGen model.
        inputs (Tensor): Input tensor of shape (batch_size, n_features).
        gamma (float): Weighting factor for the loss term.
    """
    inputs = inputs.clone().detach().requires_grad_(True)  # enable gradient tracking
    output = model(inputs)  # shape: (batch_size, n_wavelengths)

    # Compute R_top/R_s and R_p/R_s
    R_top = inputs[:, -1] # Already in meters
    R_s = inputs[:, 1] * 6.957e8 # Convert R_s from solar radii to meters
    R_p = inputs[:, 4] * 6.3781e6 # Convert R_p from Earth radii to meters

    loss = gamma * torch.sum(F.relu(outputs - (R_top/R_s)**2) + F.relu((R_p/R_s)**2 - outputs), dim=1)
    return loss.mean()  # mean over batch

O2 = [[1.26,1.28], [0.76,0.77], [0.685, 0.695]]
CO2 = [[1.42, 1.45], [1.94, 1.98]]
CH4 = [[1.6, 1.85], [1.3, 1.45], [1.1, 1.2]]
H2O = [[1.35, 1.45], [1.8, 1.95], [1.11, 1.15]]
O3 = [[0.45, 0.75]]
N2O = [[1.51, 1.535], [1.66, 1.71], [1.76, 1.79], [1.95, 2.]]
molecules = [O2, CO2, CH4, H2O, O3, N2O]
# Load the wavelength grid
import numpy as np
wl = np.loadtxt('/glade/derecho/scratch/aidenz/data/HERMES_labels/wl_grid.txt')
# Loop over the molecules and create bandpass masks
bandpass_masks = []

for mol_bands in molecules:
    mask = np.zeros_like(wl, dtype=bool)
    for band in mol_bands:
        mask |= (wl >= band[0]) & (wl <= band[1])
    bandpass_masks.append(mask)

def L_phys_chem(model, inputs, gamma, bandpass_masks = bandpass_masks, molecule_indices=[7,8,9,10,11,12]):
    """
    Computes the chemistry-based gradient loss.

    Args:


    """
    inputs = inputs.clone().detach().requires_grad_(True)  # enable gradient tracking
    output = model(inputs)  # shape: (batch_size, n_wavelengths)

    loss = 0.0
    for i, idx in enumerate(molecule_indices):
        mask = bandpass_masks[i]  # shape: (n_wavelengths,)
        delta_feature = output[:, mask].mean(dim=1)  # shape: (batch_size,)

        # Compute d(delta_feature)/dX_i
        grad = torch.autograd.grad(
            outputs=delta_feature,
            inputs=inputs,
            grad_outputs=torch.ones_like(delta_feature),
            create_graph=True,
            retain_graph=True,
        )[0]  # shape: (batch_size, 13)

        ddelta_dxi = grad[:, idx]  # shape: (batch_size,)
        penalty = F.relu(-ddelta_dxi)  # penalize negative gradients
        loss += penalty.mean()  # mean over batch

    return gamma * loss
