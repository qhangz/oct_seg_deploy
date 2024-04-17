import torch.utils.data as data
import os
import json
import platform
import pandas as pd
from PIL import Image
import numpy as np
from model.utils import split_train_test_valid

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path):

    img = Image.open(path)

    return img


# get the image list pairs
def get_imgs_list(data_dir, target1_dir, target2_dir, mode):
    '''

    :param data_dir:
    :param target_dir:
    :param mode:
    :return:
    '''
    train, valid, test = split_train_test_valid(data_dir, target1_dir, target2_dir)


    if mode == 'train':
        for data in train:
            name1=os.path.basename(data[0]).split('.')[0]
            name2=os.path.basename(data[1]).split('.')[0]
            name3=os.path.basename(data[2]).split('.')[0]
            #print(name1,"   ",name2,"   ",name3,flush=True)
            #把灰度值加在这里
            assert name1==name2==name3


        return train
    elif mode == 'valid':
        for data in valid:
            name1=os.path.basename(data[0]).split('.')[0]
            name2=os.path.basename(data[1]).split('.')[0]
            name3=os.path.basename(data[2]).split('.')[0]
            #把灰度值加在这里
            #print(name1,"   ",name2,"   ",name3,flush=True)
            assert name1==name2==name3
             #   print('----------------------------------')
        return valid
    elif mode == 'test':	
        return test


# dataset that supports one input image, one target image
class DataFolder(data.Dataset):
    def __init__(self, data_dir, target1_dir, target2_dir, mode, data_transform=None, loader=img_loader):
        super(DataFolder, self).__init__()
        self.mode = mode
        self.img_list = get_imgs_list(data_dir, target1_dir, target2_dir, self.mode)
        if len(self.img_list) == 0:
            raise (RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        #self.num_channels = num_channels
        self.loader = loader

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

class blood_Dataset(data.Dataset):
    def __init__(self, img_dir, json_dir, csv_dir, mode, data_transform=None, loader=img_loader):
        super(blood_Dataset, self).__init__()
        with open(json_dir) as f:
            dicts = json.load(f)
        self.dicts = dicts
        self.mode = mode
        self.data_transform = data_transform
        self.csv_dir = csv_dir
        self.loader = loader
        self.img_dir = img_dir
        self._preprocess()

    def _preprocess(self):
        if self.mode == 'train':
            imgs_list = self.dicts['train_x']
        elif self.mode == 'val':
            imgs_list = self.dicts['val_x']
        else:
            imgs_list = self.dicts['test_x']
        if platform.system() == 'Windows':
            imgs_list = [os.path.join('D:/oct/imgs', os.path.basename(path)) for path in imgs_list]
        else:
            imgs_list = [os.path.join(self.img_dir, os.path.basename(path)) for path in imgs_list]

        blood_gray = []
        df = pd.read_csv(self.csv_dir, index_col=0)
        for path in imgs_list:
            name = os.path.basename(path)
            gray = int(df.loc[name]['gray_value'])
            blood_gray.append(gray)
        imgs_list = [self.loader(path) for path in imgs_list]
        self._item = list(zip(imgs_list, blood_gray))

    def __getitem__(self, index):
        data = self._item[index]
        img = [data[0]]

        if self.data_transform is not None:
            img = self.data_transform(img)
        img = list(img)
        gray = data[1]
        img.append(gray)

        return tuple(img)

    def __len__(self):
        return len(self._item)
