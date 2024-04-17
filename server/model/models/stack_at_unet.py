import torch
import torch.nn as nn
from model.models.at_unet import AttU_Net


class Stack_Attention_Unet(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Stack_Attention_Unet, self).__init__()
        self.at_unet1 = AttU_Net(input_channels, out_channels)
        self.at_unet2 = AttU_Net(out_channels, out_channels)

    def forward(self, x):
        y1 = self.at_unet1(x)
        y2 = self.at_unet2(y1)

        return y1, y2
