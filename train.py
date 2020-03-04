"""
@description: 执行训练
"""


"""
import
"""
from config import ConfigTrain
import utils
from os.path import join as pjoin
import pandas as pd
import numpy as np
import cv2
import torch
import time
from tqdm import tqdm
import math
import os


"""
main
"""
if __name__ == '__main__':
    cfg = ConfigTrain()
    print('Pick device: ', cfg.DEVICE)
    device = torch.device(cfg.DEVICE)

    # 网络
    print('Generating net: ', cfg.NET_NAME)
    net = utils.create_net(3, cfg.NUM_CLASSES, net_name=cfg.NET_NAME)
    if cfg.PRETRAIN:  # 加载预训练权重
        print('Load pretrain weights: ', cfg.PRETRAINED_WEIGHTS)
        net.load_state_dict(torch.load(cfg.PRETRAINED_WEIGHTS, map_location='cpu'))
    net.to(device)
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.BASE_LR) 

    # 训练数据生成器
    print('Preparing trin data... batch_size: {}, image_size: {}, crop_offset: {}'.format(cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.HEIGHT_CROP_OFFSET))
    df_train = pd.read_csv(pjoin(cfg.DATA_LIST_ROOT, 'train.csv'))
    train_data_generator = utils.train_data_generator(cfg.IMAGE_ROOT, np.array(df_train['image']),
                                                cfg.LABEL_ROOT, np.array(df_train['label']),
                                                cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.HEIGHT_CROP_OFFSET)

    print('Preparing val data... batch_size: {}, image_size: {}, crop_offset: {}'.format(cfg.VAL_BATCH_SIZE, cfg.IMAGE_SIZE, cfg.HEIGHT_CROP_OFFSET))
    # 验证数据
    df_val = pd.read_csv(pjoin(cfg.DATA_LIST_ROOT, 'val.csv'))
    val_data_generator = utils.val_data_generator(cfg.IMAGE_ROOT, np.array(df_val['image']),
                                                cfg.LABEL_ROOT, np.array(df_val['label']),
                                                cfg.VAL_BATCH_SIZE, cfg.IMAGE_SIZE, cfg.HEIGHT_CROP_OFFSET)

    # 训练
    print('Let us train ...')
    log_iters = 1  # 多少次迭代打印一次log
    epoch_size = int(len(df_train['image']) / cfg.BATCH_SIZE)  # 一个轮次包含的迭代次数
    for epoch in range(cfg.EPOCH_BEGIN, cfg.EPOCH_NUM):
        if cfg.DEVICE.find('cuda') != -1:
            torch.cuda.empty_cache() # 回收缓存的显存
        epoch_loss = 0.0
        # epoch_miou = 0.0
        # last_epoch_miou = 0.0
        prev_time = time.time()
        epoch_start = prev_time
        for iteration in range(1 , epoch_size + 1):
            images, labels, images_filename = next(train_data_generator)
            images = images.to(device)
            labels = labels.to(device)

            lr = utils.ajust_learning_rate(optimizer, cfg.LR_STRATEGY, epoch, iteration-1, epoch_size)

            predicts = net(images)

            optimizer.zero_grad()

            loss, mean_iou = utils.create_loss(predicts, labels, cfg.NUM_CLASSES, cal_miou=False)

            epoch_loss += loss.item()
            # epoch_miou += mean_iou.item()

            left_seconds = (time.time() - prev_time) * (epoch_size-iteration)
            left_hours = left_seconds // 3600
            left_seconds = left_seconds - left_hours * 3600
            left_minutes = left_seconds // 60
            left_seconds = left_seconds - left_minutes * 60
            # print("[Epoch-%d/%d Iter-%d/%d] LR: %.4f: iter loss: %.3f, iter iou: %.3f, epoch loss: %.3f, epoch iou: %.3f,  time left: %d:%d:%d"
            #     % (epoch, cfg.EPOCH_NUM-1, iteration, epoch_size, lr, loss.item(), mean_iou.item(), epoch_loss / iteration, epoch_miou / iteration, left_hours, left_minutes, left_seconds))
            print("[Epoch-%d/%d Iter-%d/%d] LR: %.4f: iter loss: %.3f, epoch loss: %.3f,  time left: %d:%d:%d"
                % (epoch, cfg.EPOCH_NUM-1, iteration, epoch_size, lr, loss.item(), epoch_loss / iteration, left_hours, left_minutes, left_seconds))
            prev_time = time.time()

            # if mean_iou.item() < last_epoch_miou * cfg.SUSPICIOUS_RATE:
            #     with open(cfg.LOG_SUSPICIOUS_FILES, 'a+') as f:
            #         for filename in images_filename:
            #             f.write("{}\n".format(filename))
            #         f.flush()

            # last_epoch_miou = epoch_miou / iteration

            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / iteration
        # epoch_miou = epoch_miou / iteration

        # 好不容易训练完，先保存一下，免得验证时出异常，反正验证还可以通过手工进行
        # tmp_weight_file = "weights_ep_%d_%.3f_%.3f.pth" % (epoch, epoch_loss, epoch_miou)
        tmp_weight_file = "weights_ep_%d_%.3f.pth" % (epoch, epoch_loss)
        torch.save(net.state_dict(), pjoin(cfg.WEIGHTS_SAVE_ROOT, tmp_weight_file))

        # 验证
        if cfg.DEVICE.find('cuda') != -1:
            torch.cuda.empty_cache() # 回收缓存的显存
        val_loss = 0.0
        val_miou = 0.0
        val_iter_size = math.ceil(len(df_val['image']) / cfg.VAL_BATCH_SIZE)  # 遍历一次的循环次数
        iteration = 0
        while True:
            images, labels, images_filename = next(val_data_generator)
            if images is None:  # 遍历已结束
                break
            images = images.to(device)
            labels = labels.to(device)
            predicts = net(images)
            loss, mean_iou = utils.create_loss(predicts, labels, cfg.NUM_CLASSES, cal_miou=True)
            val_loss += loss.item()
            val_miou += mean_iou.item()
            iteration += 1
            print("[Iter-%d/%d] iter loss: %.3f, iter iou: %.3f, val loss: %.3f, val miou: %.3f"
                % (iteration, val_iter_size, loss.item(), mean_iou.item(), val_loss / iteration, val_miou / iteration))

            if len(images_filename) < cfg.VAL_BATCH_SIZE:  # 遍历已结束
                break
        val_loss = val_loss / iteration
        val_miou = val_miou / iteration

        # print("[Epoch-%d] epoch loss: %.3f, epoch iou: %.3f, val loss: %.3f, val miou: %.3f, time cost: %.3f s"
        #     % (epoch, epoch_loss, epoch_miou, val_loss, val_miou, time.time() - epoch_start))
        print("[Epoch-%d] epoch loss: %.3f, val loss: %.3f, val miou: %.3f, time cost: %.3f s"
            % (epoch, epoch_loss, val_loss, val_miou, time.time() - epoch_start))
        # 重命名模型权重文件
        # weight_file = "weights_ep_%d_%.3f_%.3f_%.3f_%.3f.pth" % (epoch, epoch_loss, epoch_miou, val_loss, val_miou)
        weight_file = "weights_ep_%d_%.3f_%.3f_%.3f.pth" % (epoch, epoch_loss, val_loss, val_miou)
        os.rename(pjoin(cfg.WEIGHTS_SAVE_ROOT, tmp_weight_file), pjoin(cfg.WEIGHTS_SAVE_ROOT, weight_file))
    

