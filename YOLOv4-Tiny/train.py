import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from nets.loss import yolo_loss
from nets.yolo_model import yolo_body
from utils.utils import LossHistory, ModelCheckpoint
from utils.dataMaker import data_generator

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    # 初始化参数
    eager = False  # 是否使用eager模式训练
    annotation_path = 'data/2007_train.txt'  # 获得训练图片路径和其对应的标签
    log_dir = 'logs/'  # 训练后的模型保存的位置，保存在logs文件夹里面
    classes_path = 'data/voc_classes.txt'
    anchors_path = 'data/yolo_anchors.txt'
    weights_path = 'data/yolov4_tiny_weights_coco.h5'
    input_shape = (416, 416)  # 一般在416x416和608x608选择
    normalize = False  # 是否对损失进行归一化，用于改变loss的大小
    mosaic = False
    Cosine_scheduler = False  # 余弦退火学习率
    label_smoothing = 0  # 标签平滑 0.01以下一般 如0.01、0.005
    regularization = True  # 在eager模式下是否进行正则化

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    # 创建yolo模型与模型加载
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv4-Tiny model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 2, num_classes)
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # y_true为13,13,3,85
    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], num_anchors // 2, num_classes + 5)) for l in range(2)]

    # 在这个地方设置损失，将网络的输出结果传入loss 函数，把整个模型的输出作为loss
    loss_input = [*model_body.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing, 'normalize': normalize})(loss_input)
    model = Model([model_body.input, *y_true], model_loss)

    # -------------------------------------------------------------------------------#
    #  训练参数的设置
    #  logging表示tensorboard的保存地址
    #  checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #  reduce_lr用于设置学习率下降的方式
    #  early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory(log_dir)
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    freeze_layers = 60
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#

    if True:
        Init_epoch = 0
        Freeze_epoch = 50
        batch_size = 2
        learning_rate_base = 1e-3

        epoch_size = num_train // batch_size
        epoch_size_val = num_val // batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic,
                                 random=True, eager=False),
                  steps_per_epoch=epoch_size,
                  validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                                 mosaic=False, random=False, eager=False),
                  validation_steps=epoch_size_val,
                  epochs=Freeze_epoch,
                  initial_epoch=Init_epoch,
                  callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])

    for i in range(freeze_layers): model_body.layers[i].trainable = True  # 解冻

    # 解冻后训练
    if True:
        Freeze_epoch = 50
        Epoch = 100
        batch_size = 32
        learning_rate_base = 1e-4

        epoch_size = num_train // batch_size
        epoch_size_val = num_val // batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic,
                                 random=True, eager=False),
                  steps_per_epoch=epoch_size,
                  validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                                 mosaic=False, random=False, eager=False),
                  validation_steps=epoch_size_val,
                  epochs=Epoch,
                  initial_epoch=Freeze_epoch,
                  callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
