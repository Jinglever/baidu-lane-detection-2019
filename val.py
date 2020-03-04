"""
@description: 对复赛的数据执行验证
"""


"""
import
"""
from config import ConfigVal
import utils
from os.path import join as pjoin
import pandas as pd
import numpy as np
import cv2
import torch
import time
import math


"""
main
"""
if __name__ == '__main__':
    cfg = ConfigVal()
    print('Pick device: ', cfg.DEVICE)
    device = torch.device(cfg.DEVICE)

    # 网络
    print('Generating net: ', cfg.NET_NAME)
    net = utils.create_net(3, cfg.NUM_CLASSES, net_name=cfg.NET_NAME)

    # 加载用于验证的权重
    print('Load weights: ', cfg.VAL_WEIGHTS)
    net.load_state_dict(torch.load(cfg.VAL_WEIGHTS, map_location='cpu'))
    net.to(device)

    net.eval()

    # 数据生成器
    df_val = pd.read_csv(pjoin(cfg.DATA_LIST_ROOT, 'val.csv'))
    val_data_generator = utils.val_data_generator(cfg.IMAGE_ROOT, np.array(df_val['image']),
                                                cfg.LABEL_ROOT, np.array(df_val['label']),
                                                cfg.VAL_BATCH_SIZE, cfg.IMAGE_SIZE, cfg.HEIGHT_CROP_OFFSET)

    # 验证
    print('Let us val ...')
    start_time = time.time()
    val_loss = 0.0
    val_miou = 0.0
    val_iter_size = math.ceil(len(df_val['image']) / cfg.VAL_BATCH_SIZE)  # 遍历一次的循环次数
    iteration = 0
    while True:
        if cfg.DEVICE.find('cuda') != -1:
            torch.cuda.empty_cache() # 回收缓存的显存
        images, labels, images_filename = next(val_data_generator)
        if images is None:  # 遍历已结束
            break
        images = images.to(device)
        labels = labels.to(device)
        predicts = net(images)
        loss, mean_iou = utils.create_loss(predicts, labels, cfg.NUM_CLASSES)
        val_loss += loss.item()
        val_miou += mean_iou.item()
        iteration += 1
        print("[Iter-%d/%d] iter loss: %.3f, iter iou: %.3f, val loss: %.3f, val miou: %.3f"
            % (iteration, val_iter_size, loss.item(), mean_iou.item(), val_loss / iteration, val_miou / iteration))
        
        if len(images_filename) < cfg.VAL_BATCH_SIZE:  # 遍历已结束
            break
    val_loss = val_loss / iteration
    val_miou = val_miou / iteration
    print("val loss: %.3f, val miou: %.3f, time cost: %.3f s"
        % (val_loss, val_miou, time.time() - start_time))

    print('Done')
