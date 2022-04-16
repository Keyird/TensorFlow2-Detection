from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Reshape

def get_rpn(base_layers, num_anchors):
    """
    创建建议框网络,该网络结果会对先验框进行调整获得建议框
    :param base_layers: 共享特征层：feature maps
    :param num_anchors: 先验框数量：9
    :return: 候选区域位置，类别信息
    """
    # 利用一个512通道的3x3卷积进行特征整合
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.02),
               name='rpn_conv1')(base_layers)

    # 利用一个1x1卷积调整通道数，获得预测结果
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.02),
                     name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer=RandomNormal(stddev=0.02),
                    name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr]
