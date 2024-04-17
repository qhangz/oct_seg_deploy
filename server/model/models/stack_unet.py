import torch
import torch.nn as nn
from model.models.unet import U_Net, Simple_U_Net, Simple_U_NetV2


class Stack_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Stack_UNet, self).__init__()
        self.unet1 = U_Net(img_ch, output_ch)
        self.unet2 = Simple_U_Net(output_ch + 64, output_ch)

    def forward(self, x):
        d1, y1 = self.unet1(x)  # d:64
        d1 = torch.cat((d1, y1), dim=1)
        y2 = self.unet2(d1)
        return y1, y2


class Stack_UNetV2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Stack_UNetV2, self).__init__()
        self.unet1 = U_Net(img_ch, output_ch)
        self.unet2 = Simple_U_NetV2(output_ch + 64, output_ch)

    def forward(self, x):
        d1, y1 = self.unet1(x)  # d:64
        d1 = torch.cat((d1, y1), dim=1)
        y2, y3 = self.unet2(d1)

        return y1, y2, y3  # 一阶段分割、预测、二阶段分割
