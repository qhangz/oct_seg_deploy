import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from PIL import Image
import torch.utils.data
import numpy as np
from model.models.stack_unet import Stack_UNet
from model.bin.my_transforms import get_transforms
from collections import OrderedDict
import os
import time

import cv2
import numpy as np


class Options:
    def __init__(self):
        self.dataset = 'oct'
        # --- models hyper-parameters --- #
        self.model = dict()
        self.model['in_c'] = 1  # input channel
        self.model['out_c'] = 1  # output channel
        self.model['n_layers'] = 6  # number of layers in a block
        self.model['growth_rate'] = 24  # growth_rate
        self.model['drop_rate'] = 0.1
        self.model['compress_ratio'] = 0.5
        self.model['dilations'] = [1, 2, 4, 8, 16, 4, 1]  # dilation factor for each block
        self.model['is_hybrid'] = True
        self.model['layer_type'] = 'basic'

        # --- data transform --- #
        self.transform = dict()
        # defined in parse function

        # --- test parameters --- #
        self.test = dict()
        # self.test['gpu'] = [0, ]
        self.test['blood_path'] = ''
        self.test['patch_size'] = 208
        self.test['overlap'] = 80
        self.test['model_path'] = './checkpoint_best.pth.tar'  # 加载模型

        # --- post processing --- #
        self.post = dict()
        self.post['min_area'] = 450  # minimum area for an object 20 initial
        self.post['radius'] = 4 if self.dataset == 'GlaS' else 2

        self.transform['test'] = OrderedDict()
        self.transform['test'] = {
            'to_tensor': 1,
            'normalize': [[0.020336164], [0.10145269]]
            # 'normalize': [[0.15889478], [0.20650366]] #[[0.15889478, 0.15889697, 0.15889464], [0.20650366, 0.20650636, 0.20650351]]
            # np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
        }


def seg(img_content):
    # print('seg begins.')
    start_time = time.time()

    img = img_content

    # 进行图像分割
    segmented_img = seg_generation(img)

    # 对分割后的图像进行标注
    result_img = mask_generation(img, segmented_img)

    end_time = time.time()
    # print('Segmentation and masking completed.')
    # print('共用时 %.2f 秒' % (end_time - start_time))

    # 将处理好的图像转换为字节流并返回
    return result_img, (end_time - start_time)


def seg_generation(img_content):
    # 将字节流转换为PIL.Image对象
    # img = Image.open(io.BytesIO(img_content))
    img = img_content.convert('L')  # 转换为灰度图像
    # img = img_content
    opt = Options()

    model_path = './model/checkpoint_best.pth.tar'

    test_transform = get_transforms(opt.transform['test'])
    model = Stack_UNet(opt.model['in_c'], opt.model['out_c'])
    # model = nn.DataParallel(model).cuda()
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    # 移动模型到 CPU 上
    # model = model.cpu()

    # ----- load trained model ----- #
    # print("=> loading trained model")
    # print(os.getcwd())
    # print(model_path)
    best_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(best_checkpoint['state_dict'])

    # switch to evaluate mode
    model.eval()
    # print("=> Seg begins:")

    input = test_transform((img,))[0].unsqueeze(0).cpu()

    # print('\tComputing output probability maps...')
    prob1_maps, prob2_maps = get_probmaps(input, model)

    pred2 = (prob2_maps > 0.5)
    final_pred = Image.fromarray((pred2 * 255).astype(np.uint8))

    # print('Finish segmentation.')

    return final_pred


def mask_generation(origin_img, segmented_img):
    # print('Masking begins.')
    # 处理原始图像
    origin_img = cv2.cvtColor(np.array(origin_img), cv2.COLOR_RGB2BGR)

    # 处理分割后的图像
    seg_img = np.array(segmented_img)
    # 转换为OpenCV格式的图像
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(seg_img, cv2.COLOR_BGR2HSV)
    Lower = np.array([0, 0, 200])  # 要识别白色区域的下限
    Upper = np.array([180, 30, 255])  # 要识别白色区域的上限
    mask = cv2.inRange(hsv, Lower, Upper)

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 将轮廓绘制在原始图像上
    contour_img = origin_img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    # print('Finish masking.')
    # 将处理好的图像转换为字节流并返回
    _, buffer = cv2.imencode('.png', contour_img)
    contour_img_bytes = buffer.tobytes()

    return contour_img_bytes


def get_probmaps(input, model):
    with torch.no_grad():
        output1, output2 = model(input)
    prob1_maps = output1.squeeze().sigmoid().cpu().numpy()
    prob2_maps = output2.squeeze().sigmoid().cpu().numpy()
    return prob1_maps, prob2_maps
