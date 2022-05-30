import tensorflow as tf
from tensorflow.keras import backend as K

"""
yolo_body输出的是原始值，需要将网络输出值进行解析和校正，才能获得目标框的真实值
"""

def yolo_eval(yolo_outputs,  # 即yolo_model中yolo_body的输出
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              eager=False,
              letterbox_image=True):
    """
    图片预测，包括后处理
    """

    if eager:
        image_shape = K.reshape(yolo_outputs[-1], [-1])
        num_layers = len(yolo_outputs) - 1
    else:
        # 获得特征层的数量，有效特征层的数量为3
        num_layers = len(yolo_outputs)

    # 13x13的特征层对应的anchor是[81,82], [135,169], [344,319]
    # 26x26的特征层对应的anchor是[23,27], [37,58], [81,82]
    anchor_mask = [[3, 4, 5], [1, 2, 3]]

    # 这里获得的是输入图片的大小，一般是416x416
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    # 对每个特征层进行处理
    for l in range(num_layers):
        # 将获得yolo_outputs的转换成[y_min, x_min, y_max, x_max]格式
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape, letterbox_image)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # 将每个特征层的结果进行堆叠
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
        # 取出所有box_scores >= score_threshold的框，过滤掉得分低的框
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 非极大抑制
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # 获取非极大抑制后的结果, 下列三个分别是: 框的位置，得分与种类
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    """
    将yolo_head获得的真实边界框信息[x,y,w,h]转成固定格式：[y_min, x_min, y_max, x_max]
    """
    # box_xy: -1,13,13,3,2;  box_wh: -1,13,13,3,2; box_confidence: -1,13,13,3,1;  box_class_probs: -1,13,13,3,80
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)

    # 在图像传入网络预测前会进行letterbox_image给图像周围添加灰条, 因此生成的box_xy, box_wh是相对于有灰条的图像的,我们需要对齐进行修改，去除灰条的部分。
    # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    if letterbox_image:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        boxes = K.concatenate([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ])

    # 获得最终得分和框的位置
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    解析yolov4-tiny网络的直接输出结果，将每个层级的预测结果调整成真实值[x,y,w,h]
    """
    num_anchors = len(anchors)
    # feats = tf.convert_to_tensor(feats)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格, (13, 13, 1, 2)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])

    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # 将预测结果调整成(batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值：box_xy对应框的中心点，box_wh对应框的宽和高
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回grid, feats, box_xy, box_wh
    # 在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """
    当输入图片进行了灰度填充时，调整box，并进行格式转换：[x,y,w,h] -> [y_min, x_min, y_max, x_max]
    """
    # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))

    # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况,new_shape指的是宽高缩放情况
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

