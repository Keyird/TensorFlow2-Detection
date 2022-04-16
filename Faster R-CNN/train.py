import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from nets.model import get_model
from nets.loss import ProposalTargetCreator, classifier_cls_loss, classifier_smooth_l1, rpn_cls_loss, rpn_smooth_l1
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDatasets
from utils.utils import get_classes
from utils.utils_bbox import BBoxUtility
from utils.utils_fit import fit_one_epoch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print("gpus:", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":

    # 1、初始参数设置
    classes_path = 'data/voc_classes.txt'
    model_path = 'data/pre_trained_vgg.h5'
    input_shape = [600, 600]  # 输入的shape大小
    anchors_size = [128, 256, 512]

    # 2、获取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, anchors_size)

    # 3、获得图片路径和标签，读取数据集对应的txt
    train_annotation_path = 'data/2007_train.txt'
    val_annotation_path = 'data/2007_val.txt'
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 4、训练参数的设置
    callback = tf.summary.create_file_writer("logs")  # 设置回调
    loss_history = LossHistory("logs/")  # 将历史权重存入logs，防止掉电重训

    bbox_util = BBoxUtility(num_classes)  # 实例化一个边界框解码工具
    roi_helper = ProposalTargetCreator(num_classes)  # 实例化一个候选框生成器

    # 5、创建模型，并加载权重
    K.clear_session()
    model_rpn, model_all = get_model(num_classes)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_rpn.load_weights(model_path, by_name=True)
        model_all.load_weights(model_path, by_name=True)

    # 6、训练，训练分为两个阶段：冻结阶段和解冻阶段。
    # 6.1、冻结阶段：模型的主干被冻结了，特征提取网络不发生改变。主干特征提取网络特征通用，冻结训练可以加快训练速度，也可以在训练初期防止权值被破坏。
    Init_Epoch = 0            # 起始世代
    Freeze_Epoch = 50         # 冻结训练的世代
    Freeze_batch_size = 2     # batch_size
    Freeze_lr = 1e-4          # 学习率
    freeze_layers = 17        # 冻结层数
    Freeze_Train = True       # 是否进行冻结训练，默认先冻结主干网络训练后解冻训练

    if Freeze_Train:
        for i in range(freeze_layers):
            if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
                model_all.layers[i].trainable = False  # 设置不可训练，即冻结参数
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_all.layers)))

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        model_rpn.compile(
            loss={
                'classification': rpn_cls_loss(),
                'regression': rpn_smooth_l1()
            }, optimizer=Adam(lr=lr)
        )

        model_all.compile(
            loss={
                'classification': rpn_cls_loss(),
                'regression': rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes): classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes): classifier_smooth_l1(num_classes - 1)
            }, optimizer=Adam(lr=lr)
        )

        gen = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train=True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train=False).generate()

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小或者batch太大！')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          end_epoch, anchors, bbox_util, roi_helper)
            lr = lr * 0.96  # 步长随着训练轮数的增加，指数式下降
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)

    # 6.2、解冻阶段：此时模型的主干网络参数不被冻结了，特征提取网络权重会发生改变
    UnFreeze_Epoch = 100        # 训练结束epoch
    Unfreeze_batch_size = 2     # 训练依次喂入的数据量
    Unfreeze_lr = 1e-5          # 学习率

    if Freeze_Train:
        for i in range(freeze_layers):
            if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
                model_all.layers[i].trainable = True  # 解冻参数

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        model_rpn.compile(
            loss={
                'classification': rpn_cls_loss(),
                'regression': rpn_smooth_l1()
            }, optimizer=Adam(lr=lr)
        )

        model_all.compile(
            loss={
                'classification': rpn_cls_loss(),
                'regression': rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes): classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes): classifier_smooth_l1(num_classes - 1)
            }, optimizer=Adam(lr=lr)
        )

        gen = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train=True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train=False).generate()

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          end_epoch, anchors, bbox_util, roi_helper)
            lr = lr * 0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)
