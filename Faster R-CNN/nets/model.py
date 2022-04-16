from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from nets.classifier import get_vgg_classifier
from nets.rpn import get_rpn
from nets.vgg import VGG16


def get_model(num_classes, num_anchors=9):
    # 第一阶段
    inputs = Input(shape=(None, None, 3))
    base_layers = VGG16(inputs)
    rpn = get_rpn(base_layers, num_anchors)  # 获得候选区域
    model_rpn = Model(inputs, rpn)  # 构建模型

    # 第二阶段
    proposals = Input(shape=(None, 4))  # rpn生成的候选区域
    classifier = get_vgg_classifier(base_layers, proposals, 7, num_classes)  # 将共享特征层和建议框传入classifier网络
    model_all = Model([inputs, proposals], rpn + classifier)

    return model_rpn, model_all


def get_predict_model(num_classes, num_anchors=9):
    """
    用于预测阶段构建模型
    :param num_classes: 类别数
    :param num_anchors: 先验框数量
    :return: 模型
    """
    # 第一阶段：获得候选框信息
    inputs = Input(shape=(None, None, 3))
    base_layers = VGG16(inputs)  # 600,600,3 -> 37,37,512
    rpn = get_rpn(base_layers, num_anchors)  # rpn用于生成候选框信息
    model_rpn = Model(inputs, rpn + [base_layers])  # 构建模型

    # 第二阶段：对候选框进行调整，获得更精确的定位和分类结果
    feature_maps = Input(shape=(None, None, 512))   # 共享特征层输入
    proposals = Input(shape=(None, 4))  # 候选区域
    # 首先通过 ROIPooling 层获候选区域特征；然后通过全连接层对建议框进行调整，获得最终的预测框
    classifier = get_vgg_classifier(feature_maps, proposals, 7, num_classes)
    model_classifier_only = Model([feature_maps, proposals], classifier)

    return model_rpn, model_classifier_only
