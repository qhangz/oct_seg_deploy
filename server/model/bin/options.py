import os
import numpy as np
import errno
import argparse
from collections import OrderedDict
import torch


class Options:
    def __init__(self, isTrain):
        self.dataset = 'oct'
        self.isTrain = isTrain

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

        # --- training params --- #
        self.train = dict()
        self.train['data_json'] = 'D:/oct/oct_seg3/shell/data.json'
        self.train['data_dir'] = 'D:/{}'.format(self.dataset)  # path to data  D:/oct
        self.train['save_dir'] = 'D:/experiments/{:s}/1'.format(self.dataset)
        self.train['input_size'] = 224  # input size of the image
        self.train['num_epochs'] = 300  # number of training epochs
        self.train['batch_size'] = 1  # batch size
        self.train['val_overlap'] = 80  # overlap size of patches for validation
        self.train['lr'] = 0.001  # initial learning rate
        self.train['weight_decay'] = 1e-4  # weight decay
        self.train['log_interval'] = 50  # iterations to print training results
        self.train['workers'] = 0  # number of workers to load images
        # self.train['gpu'] = [0, ]  # select gpu devices
        available_gpus = torch.cuda.device_count()
        self.train['gpu'] = list(range(available_gpus))
        self.train['alpha'] = 0.5  # weight for variance term
        self.train['checkpoint_freq'] = 100  # epoch to save checkpoints
        # --- resume training --- #
        self.train['start_epoch'] = 0  # start epoch
        self.train['checkpoint'] = ''  # checkpoint to resume training or evaluation

        # --- data transform --- #
        self.transform = dict()
        # defined in parse function

        # --- test parameters --- #
        self.test = dict()
        self.test['epoch'] = 'best'
        self.test['gpu'] = [0, ]
        self.test['img_dir'] = './images/select_gray'
        self.test['label_dir'] = './images/labels'  # 测试的时候用label_instance测量
        self.test['tta'] = False
        self.test['save_flag'] = True
        self.test['blood_path'] = ''
        self.test['eval_flag'] = True
        self.test['patch_size'] = 208
        self.test['overlap'] = 80
        self.test['data_json'] = './data.json'
        self.test['data_json_dir'] = './data.json'
        self.test['save_dir'] = './save'
        self.test['model_path'] = './checkpoint_{:s}.pth.tar'.format(self.test['epoch'])  # 加载模型

        # --- post processing --- #
        self.post = dict()
        self.post['min_area'] = 450  # minimum area for an object 20 initial
        self.post['radius'] = 4 if self.dataset == 'GlaS' else 2

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'],
                                help='input batch size for training')
            parser.add_argument('--alpha', type=float, default=self.train['alpha'],
                                help='The weight for the variance term in loss')
            parser.add_argument('--epochs', type=int, default=self.train['num_epochs'],
                                help='number of epochs to train')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'],
                                help='how many batches to wait before logging training status')
            parser.add_argument('--gpu', type=list, default=self.train['gpu'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'],
                                help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'],
                                help='directory to save training results')
            parser.add_argument('--checkpoint-path', type=str, default=self.train['checkpoint'],
                                help='directory to load a checkpoint')
            args = parser.parse_args()

            self.train['batch_size'] = args.batch_size
            self.train['alpha'] = args.alpha
            self.train['num_epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpu'] = args.gpu
            self.train['checkpoint'] = args.checkpoint_path
            self.train['data_dir'] = args.data_dir
            self.train['img_dir'] = '{:s}/select_gray'.format(self.train['data_dir'])
            self.train['label_dir'] = '{:s}/mask5'.format(self.train['data_dir'])
            self.train['label2_dir'] = '{:s}/mask3'.format(self.train['data_dir'])
            # self.train['weight_map_dir'] = '{:s}/weight_maps'.format(self.train['data_dir'])

            self.train['save_dir'] = args.save_dir
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)

            # define data transforms for training
            self.transform['train'] = OrderedDict()
            self.transform['val'] = OrderedDict()

            self.transform['train'] = {
                # 'random_resize': [0.8, 1.25],
                # 'horizontal_flip': True,
                # 'random_affine': 0.3,
                # 'random_elastic': [6, 15],
                # 'random_rotation': 90,
                # 'random_crop': self.train['input_size'],
                # 'label_encoding': 1,
                'to_tensor': 1,
                'normalize': [[0.020336164], [0.10145269]]
                # [[0.15889478, 0.15889697, 0.15889464], [0.20650366, 0.20650636, 0.20650351]]
                # np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }
            self.transform['val'] = {
                # 'label_encoding': 1,
                'to_tensor': 1,
                'normalize': [[0.020336164], [0.10145269]]
                # np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }

        else:
            parser.add_argument('--epoch', type=str, default=self.test['epoch'],
                                help='select the models used for testing')
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'],
                                help='flag to save the network outputs and predictions')
            parser.add_argument('--gpu', type=list, default=self.test['gpu'], help='GPUs for training')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--label-dir', type=str, default=self.test['label_dir'], help='directory of labels')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'],
                                help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'],
                                help='train models to be evaluated')
            parser.add_argument('--data-json', type=str, default=None, help='data split configuration')
            parser.add_argument('--data-json-dir', type=str, default=self.test['data_json_dir'],
                                help='data split configuration')
            args = parser.parse_args()
            self.test['epoch'] = args.epoch
            self.test['gpu'] = args.gpu
            self.test['save_flag'] = args.save_flag
            self.test['img_dir'] = args.img_dir
            self.test['label_dir'] = args.label_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path
            # self.test['data_json_dir'] = args.data_json
            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)
            # if not os.path.exists(self.test['data_json_dir']):
            #     raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.test['data_json_dir'])

            self.transform['test'] = OrderedDict()

            self.transform['test'] = {
                'to_tensor': 1,
                'normalize': [[0.020336164], [0.10145269]]
                # 'normalize': [[0.15889478], [0.20650366]] #[[0.15889478, 0.15889697, 0.15889464], [0.20650366, 0.20650636, 0.20650351]]
                # np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        message = self._generate_message_from_options()
        file = open(filename, 'w')
        file.write(message)
        file.close()

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-' * 25)
        train_groups = ['models', 'train', 'transform']
        test_groups = ['models', 'test', 'post', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>20}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                            message += '{:s}:\n'.format(name)
                            for t_name, t_val in val.items():
                                t_val = str(t_val).replace('\n', ',\n{:22}'.format(''))
                                message += '{:>20}: {:<35}\n'.format(t_name, str(t_val))
                else:
                    for name, val in options.items():
                        message += '{:>20}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-' * 26)
        return message
