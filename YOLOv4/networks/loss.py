import tensorflow as tf
from tensorflow.keras import backend as K
from networks.ciou import box_ciou
from networks.yolov4 import yolo_head


def _smooth_labels(y_true, label_smoothing):
    """ 平滑标签 """
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def box_iou(b1, b2):
    """ 用于计算每个预测框与真实框的iou """
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 计算重合面积
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False, normalize=True):
    """ YOLOv4的loss函数 """

    # 一共有三层
    num_layers = len(anchors) // 3

    # ---------------------------------------------------------------------------------------------------#
    #   将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    #   y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # ---------------------------------------------------------------------------------------------------#
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # 得到input_shpae为416,416
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

    loss = 0
    num_pos = 0

    # -----------------------------------------------------------#
    #   取出每一张图片
    #   m的值就是batch_size
    # -----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    # ---------------------------------------------------------------------------------------------------#
    #   y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # ---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        # -----------------------------------------------------------#
        #   以第一个特征层(m,13,13,3,85)为例子
        #   取出该特征层中存在目标的点的位置。(m,13,13,3,1)
        # -----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        # -----------------------------------------------------------#
        #   取出其对应的种类(m,13,13,3,80)
        # -----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        # -----------------------------------------------------------#
        #   将yolo_outputs的特征层输出进行处理、获得四个返回值
        #   其中：
        #   grid        (13,13,1,2) 网格坐标
        #   raw_pred    (m,13,13,3,85) 尚未处理的预测结果
        #   pred_xy     (m,13,13,3,2) 解码后的中心坐标
        #   pred_wh     (m,13,13,3,2) 解码后的宽高坐标
        # -----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        # -----------------------------------------------------------#
        #   pred_box是解码后的预测的box的位置
        #   (m,13,13,3,4)
        # -----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        # -----------------------------------------------------------#
        #   找到负样本群组，第一步是创建一个数组，[]
        # -----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # -----------------------------------------------------------#
        #   对每一张图片计算ignore_mask
        # -----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            # -----------------------------------------------------------#
            #   取出n个真实框：n,4
            # -----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # -----------------------------------------------------------#
            #   计算预测框与真实框的iou
            #   pred_box    13,13,3,4 预测框的坐标
            #   true_box    n,4 真实框的坐标
            #   iou         13,13,3,n 预测框和真实框的iou
            # -----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            # -----------------------------------------------------------#
            #   best_iou    13,13,3 每个特征点与真实框的最大重合程度
            # -----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            # -----------------------------------------------------------#
            #   判断预测框和真实框的最大iou小于ignore_thresh
            #   则认为该预测框没有与之对应的真实框
            #   该操作的目的是：
            #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
            #   不适合当作负样本，所以忽略掉。
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # -----------------------------------------------------------#
        #   在这个地方进行一个循环、循环是对每一张图片进行的
        # -----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # -----------------------------------------------------------#
        #   ignore_mask用于提取出作为负样本的特征点
        #   (m,13,13,3)
        # -----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # -----------------------------------------------------------#
        #   真实框越大，比重越小，小框的比重更大。
        # -----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # -----------------------------------------------------------#
        #   计算Ciou loss
        # -----------------------------------------------------------#
        raw_true_box = y_true[l][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)  # ciou损失函数

        # ------------------------------------------------------------------------------#
        #   如果该位置本来有框，那么计算1与置信度的交叉熵
        #   如果该位置本来没有框，那么计算0与置信度的交叉熵
        #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
        #   该操作的目的是：
        #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
        #   不适合当作负样本，所以忽略掉。
        # ------------------------------------------------------------------------------#
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        location_loss = K.sum(ciou_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)
        # -----------------------------------------------------------#
        #   计算正样本数量
        # -----------------------------------------------------------#
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss += location_loss + confidence_loss + class_loss

    loss = K.expand_dims(loss, axis=-1)

    if normalize:
        loss = loss / num_pos
    else:
        loss = loss / mf
    return loss


