import torch 
import torchvision.transforms.functional as TF
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self,in_channels , out_channels):
        spuer(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1,bias = False)
            
        )
