from tensorflow.keras.layers import Concatenate, UpSampling2D
from tensorflow.keras.models import Model
from utils.utils import compose
from nets.CSPdarknet import darknet_body, DarknetConv2D, DarknetConv2D_BN_Leaky

"""
构建YOLOv4-Tiny的整体网络模型
"""

def yolo_body(inputs, num_anchors, num_classes):
    """
    构建YOLOv4-Tiny网络模型
    """
    feat1, feat2 = darknet_body(inputs)  # 骨干网络

    # 13,13,512 -> 13,13,256
    P5 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    # 13,13,256 -> 13,13,512 -> 13,13,255
    P5_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)  # 输出1

    # 13,13,256 -> 13,13,128 -> 26,26,128
    P5_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(P5)
    # 26,26,256 + 26,26,128 -> 26,26,384
    P4 = Concatenate()([P5_upsample, feat1])
    # 26,26,384 -> 26,26,256 -> 26,26,255
    P4_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)  # 输出2

    return Model(inputs, [P5_output, P4_output])
