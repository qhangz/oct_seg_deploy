# @Time : 2021/1/8 9:43
# @Author : CaoXiang
# @Project: Semantic Segmentation
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class conv_block(nn.Module):
    '''
        aggregation of conv operation
        conv-bn-relu-conv-bn-relu
        Example:
            input:(B,C,H,W)
            conv_block(C,out)
            conv_block(input)
            rerturn (B,out,H,W)
    '''

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Basic_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(Basic_Conv_Block, self).__init__()
        self.normal_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
                      dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.normal_conv(x)


class Up_Block_A(nn.Module):
    def __init__(self, in_channels, up_channels, concat_channels, out_channels, upsample_method="transpose", up=True):
        """
        :param in_channels: 指的是输入的通道数
        :param up_channels: 指的是输入上采样后的输出通道数
        :param concat_channels: 指的是concat后的通道数
        :param out_channels: 指的是整个Up_Block的输出通道数
        :param upsample_method: 上采样方法 "conv_transpose代表转置卷积，bilinear代表双线性插值"
        :param up: 代表是否进行转置卷积，转置卷积会缩小特征图尺寸，如果不进行转置卷积，那么意味着收缩通道的下采样也需要取消掉
        """
        super(Up_Block_A, self).__init__()
        self.up = up
        if self.up == False:
            self.upsample = Basic_Conv_Block(in_channels, up_channels)
        else:
            if upsample_method == "transpose":
                self.upsample = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2)
            elif upsample_method == "bilinear":
                self.upsample = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                    nn.Conv2d(in_channels, up_channels, kernel_size=1, stride=1)
                )
        self.conv1 = Basic_Conv_Block(concat_channels, out_channels)
        self.conv2 = Basic_Conv_Block(out_channels, out_channels)

    def forward(self, x, shortcut, enc_feature=None):
        x = self.upsample(x)
        # print(x.shape, shortcut.shape)
        if enc_feature is None:
            x = torch.cat([x, shortcut], dim=1)
        else:
            # print('up:', x.size(), shortcut.size(), enc_feature.size())
            x = torch.cat([x, shortcut, enc_feature], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CSE_Block(nn.Module):
    def __init__(self, inplanes, reduce_ratio=4):
        super(CSE_Block, self).__init__()
        self.iplanes = inplanes
        self.reduce_ratio = reduce_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(inplanes, inplanes // self.reduce_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(inplanes // self.reduce_ratio, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        out = input * x
        return out


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = Basic_Conv_Block(inplanes, outplanes, kernel_size=1, padding=0, dilation=dilations[0])
        self.aspp2 = Basic_Conv_Block(inplanes, outplanes, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = Basic_Conv_Block(inplanes, outplanes, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = Basic_Conv_Block(inplanes, outplanes, kernel_size=3, padding=dilations[3], dilation=dilations[3])
        self.avg_pool = nn.Sequential(
            nn.AvgPool2d((1, 1)),
            Basic_Conv_Block(inplanes, outplanes, kernel_size=1, padding=0)
        )

        self.project = nn.Sequential(
            nn.Conv2d(5 * outplanes, outplanes, 1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU())

    def forward(self, x):
        x1 = self.aspp1(x)
        # print("x1 shape:", x1.shape)
        x2 = self.aspp2(x)
        # print("x2 shape:", x2.shape)
        x3 = self.aspp3(x)
        # print("x3 shape:", x3.shape)
        x4 = self.aspp4(x)
        # print("x4 shape:", x4.shape)
        x5 = self.avg_pool(x)
        # print("x5 shape:", x5.shape)
        x5 = F.interpolate(x5, x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.project(x)
        return x


# first unet
class Unet_A(nn.Module):
    def __init__(self, base_model):
        super(Unet_A, self).__init__()
        # first encoder
        self.base_model = base_model
        self.layer1 = list(self.base_model.children())[0][:6]
        self.layer2 = list(self.base_model.children())[0][7:13]
        self.layer3 = list(self.base_model.children())[0][14:26]
        self.layer4 = list(self.base_model.children())[0][27:39]
        self.layer5 = list(self.base_model.children())[0][40:52]
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)
        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

        # first aspp
        self.aspp = ASPP(512, 512)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512)
        self.up3 = Up_Block_A(512, 256, 512, 256)
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64)

        # first out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        input = x
        # print("unetA input shape:", input.shape)
        en_x1 = self.layer1(x)
        en_x1 = self.se1(en_x1)
        pool_x1 = self.pool(en_x1)
        # print("unetA pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)
        en_x2 = self.se2(en_x2)
        pool_x2 = self.pool(en_x2)
        # print("unetA pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)
        en_x3 = self.se3(en_x3)
        pool_x3 = self.pool(en_x3)
        # print("unetA pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)
        en_x4 = self.se4(en_x4)
        pool_x4 = self.pool(en_x4)
        # print("unetA pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)
        en_x5 = self.se5(en_x5)
        # pool_x5 = self.pool(en_x5)
        # print("unetA pool_x5 shape:", pool_x5.shape)

        aspp_out = self.aspp(en_x5)
        # print("unetA aspp shape:", aspp_out.shape)

        de_x4 = self.up4(aspp_out, en_x4)
        # print("unetA de_x4 shape:", de_x4.shape)
        de_x3 = self.up3(de_x4, en_x3)
        # print("unetA de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # print("unetA de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)
        # print("unetA de_x1 shape:", de_x1.shape)

        encoder_features = [en_x1, en_x2, en_x3, en_x4]

        output = self.out_conv(de_x1)
        # output = F.sigmoid(output)
        # print("unetA output shape:", output.shape)

        return input, output, encoder_features


# second unet
class Unet_B(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_B, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        self.aspp = ASPP(512, 512)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1536, 512)
        self.up3 = Up_Block_A(512, 256, 768, 256)
        self.up2 = Up_Block_A(256, 128, 384, 128)
        self.up1 = Up_Block_A(128, 64, 192, 64)

        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

        # second out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.combine_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

    def forward(self, input, output1, encoder_features):
        enc1, enc2, enc3, enc4 = encoder_features
        input_2 = input * output1
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(input_2)  # 224,224,64
        en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)  # 112,112,128
        en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)  # 56,56,256
        en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)  # 28,28,512
        en_x4 = self.se4(en_x4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(aspp_out, en_x4, enc4)
        de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3, enc3)
        de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2, enc2)
        de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1, enc1)
        de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        output = torch.cat([output1, output2], dim=1)
        output = self.combine_conv(output)

        return output


class DoubleUnet(nn.Module):
    def __init__(self, base_model):
        super(DoubleUnet, self).__init__()
        self.base_model = base_model
        self.unet1 = Unet_A(self.base_model)
        self.unet2 = Unet_B(3, 1, 64)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        input, output1, encoder_features = self.unet1(x)
        input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, input2, encoder_features)
        return output1, output2


# if __name__ == '__main__':
#     # def count_param(model):
#     #     param_count = 0
#     #     for param in model.parameters():
#     #         param_count += param.view(-1).size()[0]
#     #     return param_count
#
#     base_model = models.vgg19_bn()
#     # print(base_model)
#     model = DoubleUnet(base_model)
#     input = torch.randn(1, 3, 224, 224)
#     output1, output2 = model(input)
#     print(output1.size(), output2.size())
#     # print(model(input).shape)
#     # print(count_param(model))
