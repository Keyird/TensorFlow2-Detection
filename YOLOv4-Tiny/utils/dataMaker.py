from utils.utils import get_random_data, get_random_data_with_Mosaic
import numpy as np

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False, random=True, eager=True):
    """ 训练数据生成器 """
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i + 4) < n:   # Mosaic数据增强
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i + 4], input_shape)
                    i = (i + 4) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                    i = (i + 1) % n
                flag = bool(1 - flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                i = (i + 1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        if eager:
            yield image_data, y_true[0], y_true[1]
        else:
            yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    读入xml文件，并输出y_true
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有两个特征层数
    num_layers = len(anchors) // 3

    # 13x13的特征层对应的anchor是[81,82], [135,169], [344,319]; 26x26的特征层对应的anchor是[23,27], [37,58], [81,82]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # 获得框的坐标和图片的大小
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    # 通过计算获得真实框的中心和宽高, 中心点(m,n,2) 宽高(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    # 将真实框归一化到小数形式
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    #   [6,2] -> [1,6,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # [n,2] -> [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算所有真实框和先验框的交并比
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是[n,] 感谢 消尽不死鸟 的提醒
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            #   找到每个真实框所属的特征层
            # -----------------------------------------------------------#
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # -----------------------------------------------------------#
                    #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                    # -----------------------------------------------------------#
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # -----------------------------------------------------------#
                    #   k指的的当前这个特征点的第k个先验框
                    # -----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    # -----------------------------------------------------------#
                    #   c指的是当前这个真实框的种类
                    # -----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    # -----------------------------------------------------------#
                    #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                    #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                    #   1代表的是置信度、80代表的是种类
                    # -----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
    return y_true
