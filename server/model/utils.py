import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2
import skimage.morphology as morph
from scipy.spatial.distance import directed_hausdorff as hausdorff


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def split_train_test_valid(data_dir, target1_dir, target2_dir):
    '''

    :param data_dir:源数据路径
    :param target_dir: ground Truth 数据路径
    :return:train,valid,test
    '''
    if os.path.exists('./data.json'):
        print('ok_exist', flush=True)
        with open('./data.json') as f:
            dicts = json.load(f)
        print(dicts)
        # dicts['train_x'] = [os.path.join('D:/oct/imgs', os.path.basename(path)) for path in dicts['train_x']]
        # dicts['train_y1'] = [os.path.join('D:/oct/mask5', os.path.basename(path)) for path in dicts['train_y1']]
        # dicts['train_y2'] = [os.path.join('D:/oct/mask3', os.path.basename(path)) for path in dicts['train_y2']]
        train = zip(dicts['train_x'], dicts['train_y1'], dicts['train_y2'])
        valid = zip(dicts['val_x'], dicts['val_y1'], dicts['val_y2'])
        test = zip(dicts['test_x'], dicts['test_y1'], dicts['test_y2'])
    else:
        json_dict = {}
        np.random.seed(0)  # 设置一个随机数以保持每次数据都是一样的
        img_list = glob.glob(os.path.join(data_dir, '*.png'))
        img_list.sort()
        img_list = np.array(img_list)
        mask1_list = []
        mask2_list = []
        for path in img_list:             
             name = os.path.basename(path).split('.')[0]             
             mask1_path = os.path.join(target1_dir, name + '.png')             
             mask2_path = os.path.join(target2_dir, name + '.png')             
             mask1_list.append(mask1_path)             
             mask2_list.append(mask2_path)
        nums = len(img_list)
        mask1_list = np.array(mask1_list)
        mask2_list = np.array(mask2_list)
        # define the ratios 6:2:2
        train_len = int(nums * 0.6)

        test_len = int(nums * 0.2)

        # split the dataframe
        idx = np.arange(nums)
        np.random.shuffle(idx)  # 将index列表打乱
        df_train_x = img_list[idx[:train_len]].tolist()
        df_train_y1 = mask1_list[idx[:train_len]].tolist()
        df_train_y2 = mask2_list[idx[:train_len]].tolist()

        df_test_x = img_list[idx[train_len:train_len + test_len]].tolist()
        df_test_y1 = mask1_list[idx[train_len:train_len + test_len]].tolist()
        df_test_y2 = mask2_list[idx[train_len:train_len + test_len]].tolist()

        df_valid_x = img_list[idx[train_len + test_len:]].tolist()  # 剩下的就是valid
        df_valid_y1 = mask1_list[idx[train_len + test_len:]].tolist()
        df_valid_y2 = mask2_list[idx[train_len + test_len:]].tolist()

        json_dict['train_x'] = df_train_x
        json_dict['train_y1'] = df_train_y1
        json_dict['train_y2'] = df_train_y2
        json_dict['val_x'] = df_valid_x
        json_dict['val_y1'] = df_valid_y1
        json_dict['val_y2'] = df_valid_y2
        json_dict['test_x'] = df_test_x
        json_dict['test_y1'] = df_test_y1
        json_dict['test_y2'] = df_test_y2
        with open('data.json', 'w') as f:
            json.dump(json_dict, f, indent=1)
        train = zip(df_train_x, df_train_y1, df_train_y2)
        test = zip(df_test_x, df_test_y1, df_test_y2)
        valid = zip(df_valid_x, df_valid_y1, df_valid_y2)
        # output
    return list(train), list(valid), list(test)


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        #print('val:{}, self.val:{}'.format(val.shape, self.val.shape),flush=True)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # 这个avg求的是每个batch的平均


def accuracy_pixel_level(output, target):
    """ Computes the accuracy during training and validation for ternary label """
    batch_size = target.shape[0]  # 1
    results = np.zeros((8,), np.float)

    for i in range(batch_size):
        pred = output[i, :, :]  # 每个样本
        label = target[i, :, :]

        # inside part
        pred_inside = pred == 1
        label_inside = label == 1
        metrics_inside = compute_pixel_level_metrics(pred_inside, label_inside)

        results += np.array(metrics_inside)

    return [value / batch_size for value in results]


def compute_pixel_level_metrics(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    # # modify
    # # 确保 pred 和 target 的值在 [0, 1] 范围内
    # pred = np.clip(pred, 0, 1)
    # target = np.clip(target, 0, 1)
    # # 取 target 的平均值作为新的 target
    # target_reshaped = np.mean(target, axis=-1)
    # tp = np.sum(pred * target_reshaped)  # true postives
    # tn = np.sum((1 - pred) * (1 - target_reshaped))  # true negatives
    # fp = np.sum(pred * (1 - target_reshaped))  # false postives
    # fn = np.sum((1 - pred) * target_reshaped)  # false negatives
    # # modify

    tp = np.sum(pred * target)  # true postives
    tn = np.sum((1 - pred) * (1 - target))  # true negatives
    fp = np.sum(pred * (1 - target))  # false postives
    fn = np.sum((1 - pred) * target)  # false negatives

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    performance = (recall + tn / (tn + fp + 1e-10)) / 2
    iou = tp / (tp + fp + fn + 1e-10)
    dice = 2 * tp / (tp + fn + tp + fp + 1e-10)


    haus = max(hausdorff(pred, target)[0], hausdorff(target, pred)[0])

    return [acc, iou, recall, precision, F1, performance, dice, haus]


def nuclei_accuracy_object_level(pred, gt):
    """ Computes the accuracy during test phase of nuclei segmentation """
    # get connected components
    pred_labeled = np.copy(pred)
    gt_labeled = np.copy(gt)
    # pred_labeled = measure.label(pred)
    # gt_labeled = measure.label(gt)
    Ns = len(np.unique(pred_labeled)) - 1  # number of detected objects
    Ng = len(np.unique(gt_labeled)) - 1  # number of ground truth objects

    TP = 0.0  # true positive
    FN = 0.0  # false negative
    dice = 0.0
    haus = 0.0
    iou = 0.0
    C = 0.0
    U = 0.0
    # pred_copy = np.copy(pred)
    count = 0.0

    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_part = pred_labeled * gt_i

        # get intersection objects numbers in pred_labeled
        obj_no = np.unique(overlap_part)
        obj_no = obj_no[obj_no != 0]

        # no intersection object
        if obj_no.size == 0:
            FN += 1
            U += np.sum(gt_i)
            continue

        # find max iou object
        max_iou = 0.0
        for k in obj_no:
            tmp_overlap_area = np.sum(overlap_part == k)
            tmp_pred = np.where(pred_labeled == k, 1, 0)  # segmented object
            tmp_iou = float(tmp_overlap_area) / (np.sum(tmp_pred) + np.sum(gt_i) - tmp_overlap_area)
            if tmp_iou > max_iou:
                max_iou = tmp_iou
                pred_i = tmp_pred
                overlap_area = tmp_overlap_area

        TP += 1
        count += 1

        # compute dice and iou
        dice += 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
        iou += float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

        # compute hausdorff distance
        seg_ind = np.argwhere(pred_i)
        gt_ind = np.argwhere(gt_i)
        haus += max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        # compute AJI
        C += overlap_area
        U += np.sum(pred_i) + np.sum(gt_i) - overlap_area

        # pred_copy[pred_i > 0] = 0
        pred_labeled[pred_i > 0] = 0  # remove the used nucleus

    # compute recall, precision, F1
    FP = Ns - TP
    recall = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    F1 = 2 * TP / (2 * TP + FP + FN + 1e-10)

    dice /= count
    iou /= count
    haus /= count

    # compute AJI
    U += np.sum(pred_labeled > 0)
    AJI = float(C) / U

    return recall, precision, F1, dice, iou, haus, AJI
    # return dice, iou, haus, AJI


def object_F1(pred, gt):
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = morph.label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = morph.label(gt, connectivity=2)
    gt_labeled = morph.remove_small_objects(gt_labeled, 3)  # remove 1 or 2 pixel noise in the image
    gt_labeled = morph.label(gt_labeled, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # show_figures((pred_labeled, gt_labeled))

    # --- compute F1 --- #
    TP = 0.0  # true positive
    FP = 0.0  # false positive
    for i in range(1, Ns + 1):
        pred_i = np.where(pred_labeled == i, 1, 0)
        img_and = np.logical_and(gt_labeled, pred_i)

        # get intersection objects in target
        overlap_parts = img_and * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((img_i, overlap_parts))

        # no intersection object
        if obj_no.size == 0:
            FP += 1
            continue

        # find max overlap object
        obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
        gt_obj = obj_no[np.argmax(obj_areas)]  # ground truth object number

        gt_obj_area = np.sum(gt_labeled == gt_obj)  # ground truth object area
        overlap_area = np.sum(overlap_parts == gt_obj)

        if float(overlap_area) / gt_obj_area >= 0.5:
            TP += 1
        else:
            FP += 1

    FN = Ng - TP  # false negative

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

    return recall, precision, F1


def split_forward(model, input, size, overlap, outchannel=3):
    '''
    split the input image for forward process
    根据overlap填充h和w使其能够完整切完，填充的地方都为0，截取算完之后非填充部分作为结果即可
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w))
    for i in range(0, h - overlap, size - overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w - overlap, size - overlap):
            c_end = j + size if j + size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch = model(input_var)

            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w
            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                         ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].cuda()

    return output


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        logits = logits.view(-1)
        gt = labels.view(-1)
        # http://geek.csdn.net/news/detail/126833
        loss = logits.clamp(min=0) - logits * gt + torch.log(1 + torch.exp(-logits.abs()))  # clamp控制下限为0
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)
        batch_size = target.size()[0]
        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(batch_size, -1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1 - pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


def rgb2hsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j] - g[i, j]) + (r[i, j] - b[i, j]))
            den = np.sqrt((r[i, j] - g[i, j]) ** 2 + (r[i, j] - b[i, j]) * (g[i, j] - b[i, j]))
            theta = float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j] + g[i, j] + r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * 3.14159265)
            I = sum / 3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H * 255
            hsi_lwpImg[i, j, 1] = S * 255
            hsi_lwpImg[i, j, 2] = I * 255
    return hsi_lwpImg
