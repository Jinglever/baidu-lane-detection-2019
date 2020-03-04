from os.path import join as pjoin
from os.path import dirname, abspath

class ConfigTrain(object):
    # 目录
    PROJECT_ROOT = dirname(abspath(__file__))
    DATA_LIST_ROOT = pjoin(PROJECT_ROOT, 'data_list')
    TRAIN_ROOT = '/root/data/LaneSeg'
    IMAGE_ROOT = pjoin(TRAIN_ROOT, 'Image_Data')
    LABEL_ROOT = pjoin(TRAIN_ROOT, 'Gray_Label')
    WEIGHTS_ROOT = pjoin(PROJECT_ROOT, 'weights')
    WEIGHTS_SAVE_ROOT = pjoin(WEIGHTS_ROOT, '1536x512_b2')
    LOG_ROOT = pjoin(PROJECT_ROOT, 'logs')

    # log文件
    LOG_SUSPICIOUS_FILES = pjoin(LOG_ROOT, 'suspicious_files.log')

    # 设备
    DEVICE = 'cuda:0'

    # 网络类型
    NET_NAME = 'unet'

    # 网络参数
    NUM_CLASSES = 8  # 8个类别
    IMAGE_SIZE = (1536, 512)  # 训练的图片的尺寸(h,w)
    HEIGHT_CROP_OFFSET = 690  # 在height方向上将原图裁掉的offset
    BATCH_SIZE = 2  # 数据批次大小
    EPOCH_NUM = 30  # 总轮次
    PRETRAIN = False  # 是否加载预训练的权重
    EPOCH_BEGIN = 0  # 接着前面的epoch训练，默认0，表示从头训练
    PRETRAINED_WEIGHTS = pjoin(WEIGHTS_ROOT, '1024x384_b4', 'weights_ep_21_0.008_0.013_0.648.pth')
    BASE_LR = 0.001  # 学习率
    LR_STRATEGY = [
        [0.001], # epoch 0
        [0.001], # epoch 1
        [0.001], # epoch 2
        [0.001, 0.0006, 0.0003, 0.0001, 0.0004, 0.0008, 0.001], # epoch 3
        [0.001, 0.0006, 0.0003, 0.0001, 0.0004, 0.0008, 0.001], # epoch 4
        [0.001, 0.0006, 0.0003, 0.0001, 0.0004, 0.0008, 0.001], # epoch 5
        [0.0004, 0.0003, 0.0002, 0.0001, 0.0002, 0.0003, 0.0004], # epoch 6
        [0.0004, 0.0003, 0.0002, 0.0001, 0.0002, 0.0003, 0.0004], # epoch 7

        [0.0003, 0.0002, 0.0001, 0.0002, 0.0003], # epoch 8
        [0.0003, 0.0002, 0.0001, 0.0002, 0.0003], # epoch 9
        [0.0003, 0.0002, 0.0001, 0.0002, 0.0003], # epoch 10
        [0.0003, 0.0002, 0.0001, 0.0002, 0.0003], # epoch 11
        [0.0003, 0.0002, 0.0001, 0.0002, 0.0003], # epoch 12

        [0.0002, 0.0001, 0.0002], # epoch 13
        [0.0002, 0.0001, 0.0002], # epoch 14
        [0.0002, 0.0001, 0.0002], # epoch 15
        [0.0002, 0.0001, 0.0002], # epoch 16
        [0.0002, 0.0001, 0.0002], # epoch 17

        [0.0001], # epoch 18
        [0.0001], # epoch 19
        [0.0001], # epoch 20
        [0.0001], # epoch 21
        [0.0001], # epoch 22
        [0.0001], # epoch 23
        [0.0001], # epoch 24
        [0.0001], # epoch 25
        [0.0001], # epoch 26
        [0.0001], # epoch 27
        [0.0001], # epoch 28
        [0.0001], # epoch 29
        [0.0001], # epoch 30
    ]
    SUSPICIOUS_RATE = 0.75  # 可疑比例：当某个iteration的miou比当前epoch_miou的可疑比例还要小的时候，记录此次iteration的训练数据索引，人工排查是否数据有问题

    VAL_BATCH_SIZE = 1  # 验证的数据批次大小

    
class ConfigInference(object):
    # 目录
    PROJECT_ROOT = dirname(abspath(__file__))
    DATA_ROOT = '/root/data/test'
    IMAGE_ROOT = pjoin(DATA_ROOT, 'TestImage')
    LABEL_ROOT = pjoin(DATA_ROOT, 'TestLabel')
    OVERLAY_ROOT = pjoin(DATA_ROOT, 'TestOverlay')
    WEIGHTS_ROOT = pjoin(PROJECT_ROOT, 'weights')
    PRETRAINED_WEIGHTS = pjoin(WEIGHTS_ROOT, '1536x512_b2', 'weights_ep_18_0.007_0.011_0.684.pth')
    LOG_ROOT = pjoin(PROJECT_ROOT, 'logs')

    # 设备
    DEVICE = 'cuda:0'

    # 网络类型
    NET_NAME = 'unet'

    # 网络参数
    NUM_CLASSES = 8  # 8个类别
    IMAGE_SIZE = (1536, 512)  # 训练的图片的尺寸(h,w)
    HEIGHT_CROP_OFFSET = 690  # 在height方向上将原图裁掉的offset
    BATCH_SIZE = 1  # 数据批次大小

    # 原图的大小
    IMAGE_SIZE_ORG = (3384, 1710)

    # 标签模式 color | gray
    LABEL_MODE = 'gray'

class ConfigVal(object):
    # 目录
    PROJECT_ROOT = dirname(abspath(__file__))
    DATA_LIST_ROOT = pjoin(PROJECT_ROOT, 'data_list')
    VAL_ROOT = '/root/data/LaneSeg'
    IMAGE_ROOT = pjoin(VAL_ROOT, 'Image_Data')
    LABEL_ROOT = pjoin(VAL_ROOT, 'Gray_Label')
    WEIGHTS_ROOT = pjoin(PROJECT_ROOT, 'weights')
    VAL_WEIGHTS = pjoin(WEIGHTS_ROOT, '1536x512_b2', 'weights_ep_18_0.007_0.011_0.684.pth')
    LOG_ROOT = pjoin(PROJECT_ROOT, 'logs')

    # 数据量
    VAL_MAX_NUM = 999  # 最多取出多少做训练集

    # 设备
    DEVICE = 'cuda:0'

    # 网络类型
    NET_NAME = 'unet'

    # 网络参数
    NUM_CLASSES = 8  # 8个类别
    IMAGE_SIZE = (1536, 512)  # 训练的图片的尺寸(h,w)
    HEIGHT_CROP_OFFSET = 690  # 在height方向上将原图裁掉的offset
    VAL_BATCH_SIZE = 1  # 数据批次大小

