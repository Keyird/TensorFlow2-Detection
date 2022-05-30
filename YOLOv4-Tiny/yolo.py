import os
import colorsys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from nets.yolov4_tiny import yolo_eval
from nets.yolo_model import yolo_body
from utils.utils import letterbox_image

class YOLO(object):
    _defaults = {
        "model_path": 'data/yolov4_tiny_weights_voc.h5',
        # "model_path": 'logs/ep055-loss8.183-val_loss9.923.h5', # 训练完后，选择加载最佳的模型
        "anchors_path": 'data/yolo_anchors.txt',
        "classes_path": 'data/voc_classes.txt',
        "score": 0.5,
        "iou": 0.3,
        "max_boxes": 100,
        "model_image_size": (416, 416),  # 显存比较小可以使用416x416,显存较大可用608x608
        "letterbox_image": False,  # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize, 测试发现直接resize的效果更好
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        """ 初始化 yolov4-tiny """
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        """ 获得所有的分类 """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """ 获得所有的先验框 """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算先验框的数量和种类的数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
        self.yolo_model.load_weights(self.model_path)
        # self.yolo_model.save_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # 在yolo_eval函数中，对预测结果进行后处理(解码、非极大抑制、门限筛选)
        self.input_image_shape = Input([2, ], batch_size=1)
        inputs = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                         arguments={'anchors': self.anchors,
                                    'num_classes': len(self.class_names),
                                    'image_shape': self.model_image_size,
                                    'score_threshold': self.score,
                                    'eager': True,
                                    'max_boxes': self.max_boxes,
                                    'letterbox_image': self.letterbox_image})(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    def detect_image(self, image):
        """ 检测一帧图像 """
        # 在这里将图像转换成RGB图像，防止灰度图在预测时报错
        image = image.convert('RGB')

        # 给图像增加灰条，实现不失真的resize; 也可以直接resize进行识别
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        else:
            boxed_image = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.

        # 添加上batch_size维度
        image_data = np.expand_dims(image_data, 0)

        # 将图像输入网络当中进行预测
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        #
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # 设置字体
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
