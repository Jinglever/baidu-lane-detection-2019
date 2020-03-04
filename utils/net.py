"""
@description: 跟网络模型有关的函数库
"""


"""
import
"""
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import nets  # 本地
import utils  # 本地
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_net(in_channels, out_channels, net_name='unet'):
    """
    创建网络
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    # :param net_name: 网络类型，可选 unet | unet_resnet18/34/50/101/152 |unet_resnext50_32x4d | deeplabv3p
    :param net_name: 网络类型，可选 unet | unet_resnet34
    """
    if net_name == 'unet':
        net = nets.unet(in_channels, out_channels, same=True)
    elif net_name == 'unet_resnet34':
        net = nets.unet_resnet('resnet34', in_channels, out_channels)
    elif net_name == 'unet_resnet50':
        net = nets.unet_resnet('resnet50', in_channels, out_channels)
    elif net_name == 'resnext50_32x4d':
        net = nets.unet_resnet('resnext50_32x4d', in_channels, out_channels)
    else:
        raise ValueError('Not supported net_name: {}'.format(net_name))

    return net

def create_loss(predicts: torch.Tensor, labels: torch.Tensor, num_classes, cal_miou=True):
    """
    创建loss
    @param predicts: shape=(n, c, h, w)
    @param labels: shape=(n, h, w) or shape=(n, 1, h, w)
    @param num_classes: int should equal to channels of predicts
    @return: loss, mean_iou
    """
    # permute to (n, h, w, c)
    predicts = predicts.permute((0, 2, 3, 1))
    # reshape to (-1, num_classes)  每个像素在每种分类上都有一个概率
    predicts = predicts.reshape((-1, num_classes))
    # BCE with DICE
    bce_loss = F.cross_entropy(predicts, labels.flatten(), reduction='mean')  # 函数内会自动做softmax
    # 将labels做one_hot处理，得到的形状跟predicts相同
    labels_one_hot = utils.make_one_hot(labels.reshape((-1, 1)), num_classes)
    dice_loss = utils.DiceLoss()(predicts, labels_one_hot.to(labels.device))  # torch没有原生的，从老师给的代码里拿过来用
    loss = bce_loss + dice_loss
    if cal_miou:
        ious = compute_iou(predicts, labels.reshape((-1, 1)), num_classes)
        miou = np.nanmean(ious.numpy())
    else:
        miou = None
    return loss, miou

def compute_iou(predicts, labels, num_classes):
    """
    计算iou
    @param predicts: shape=(-1, classes)
    @param labels: shape=(-1, 1)
    """
    ious = torch.zeros(num_classes)
    predicts = F.softmax(predicts, dim=1)
    predicts = torch.argmax(predicts, dim=1, keepdim=True)
    for i in range(num_classes):
        intersect = torch.sum((predicts == i) * (labels == i))
        area = torch.sum(predicts == i) + torch.sum(labels == i) - intersect
        if area == 0 and intersect == 0:
            ious[i] = np.nan  # 忽略这种iou
        else:
            ious[i] = intersect.float() / area.float()
    return ious

def ajust_learning_rate(optimizer, lr_strategy, epoch, iteration, epoch_size):
    """
    根据给定的策略调整学习率
    @param optimizer: 优化器
    @param lr_strategy: 策略，一个二维数组，第一维度对应epoch，第二维度表示在一个epoch内，若干阶段的学习率
    @param epoch: 当前在第几号epoch
    @param iteration: 当前epoch内的第几次迭代
    @param epoch_size: 当前epoch的总迭代次数
    """
    assert epoch < len(lr_strategy), 'lr strategy unconvering all epoch'
    batch = epoch_size // len(lr_strategy[epoch])
    lr = lr_strategy[epoch][-1]
    for i in range(len(lr_strategy[epoch])):
        if iteration < (i + 1) * batch:
            lr = lr_strategy[epoch][i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            break
    return lr

