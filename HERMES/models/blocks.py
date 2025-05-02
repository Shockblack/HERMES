#--------------------------------------------------------------
# Filename: blocks.py
#
# Programmer: Aiden Zelakiewicz (asz39@cornell.edu)
#
# Dependencies: PyTorch
#--------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class convBlock(nn.Module):
    """
    A block layer that applies a series of convolutional and activation layers.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, final_layer: bool = False) -> None:
        """
        Initialize the BlockLayer.
        Increases size of the input tensor by using interpolation and a convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Default is 3.
            stride (int): Stride of the convolution. Default is 1.
            padding (int): Padding for the convolution. Default is 1.
        """
        super(convBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.final_layer = final_layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: Tensor) -> Tensor: 
        """
        Forward pass through the block layer.

        Args:
            x (Tensor): Input tensor.
        """
        inputs = x # 64
        x = F.interpolate(x, scale_factor=2) # 128
        x = self.conv(x) # Calculate this (x,)
        x = self.bn(x)
        if not self.final_layer:
            x = self.dropout(x)
        x = self.relu(x)
        x = torch.cat((inputs, x), dim=1)
        return x
