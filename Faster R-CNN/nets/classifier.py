import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Layer, TimeDistributed
from nets.vgg import vgg_classifier_layers


class RoiPoolingConv(Layer):
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        input_shape2 = input_shape[1]
        return None, input_shape2[1], self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)
        # 共享特征层：batch_size, 38, 38, 1024
        feature_map = x[0]

        # 建议框：batch_size, num_rois, 4
        rois = x[1]

        # 建议框数量，batch_size大小
        num_rois = tf.shape(rois)[1]
        batch_size = tf.shape(rois)[0]

        # 生成建议框序号信息，用于在进行crop_and_resize时，帮助建议框找到对应的共享特征层
        box_index = tf.expand_dims(tf.range(0, batch_size), 1)
        box_index = tf.tile(box_index, (1, num_rois))
        box_index = tf.reshape(box_index, [-1])
        rs = tf.image.crop_and_resize(feature_map, tf.reshape(rois, [-1, 4]), box_index,
                                      (self.pool_size, self.pool_size))

        # (batch_size, num_rois, 14, 14, 1024)
        final_output = K.reshape(rs, (batch_size, num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output


def get_vgg_classifier(base_layers, proposals, roi_size, num_classes=21):
    """
    将共享特征层和建议框传入classifier网络,该网络结果会对建议框进行调整获得预测框。
    :param base_layers:共享特征层
    :param input_rois:输入ROI
    :param roi_size:ROI尺寸
    :param num_classes:类别数
    :return: 目标类别和边界框位置信息
    """
    # 将特征共享层 base_layers 和 proposals 同时送入 ROIPooling 层，提取 proposal feature maps
    # [batch_size, 37, 37, 512] -> [batch_size, num_rois, 7, 7, 512]
    proposal_feature_maps = RoiPoolingConv(roi_size)([base_layers, proposals])

    # 将得到的推荐区域特征图送入全连接网络（VGG16的全连接部分）
    # [batch_size, num_rois, 7, 7, 512] -> [batch_size, num_rois, 4096]
    out = vgg_classifier_layers(proposal_feature_maps)

    # 1、类别预测
    # [batch_size, num_rois, 4096] -> ]batch_size, num_rois, num_classes]
    out_class = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02)),
                                name='dense_class_{}'.format(num_classes))(out)

    # 2、边界框的二次精确回归
    # batch_size, num_rois, 4096 -> batch_size, num_rois, 4 * (num_classes-1)
    out_regr = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=RandomNormal(stddev=0.02)),
                                name='dense_regress_{}'.format(num_classes))(out)
    return [out_class, out_regr]
