import cv2
import time
import numpy as np
from PIL import Image
from yolo import YOLO
import tensorflow as tf
import os

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 根据自己的视频路径设置
capture = cv2.VideoCapture("D:\\Project\\faster-rcnn-tf2\\1.mp4")
fps = 0.0
yolov4_tiny = YOLO()

while (True):
    t1 = time.time()
    # 读取某一帧
    ref, frame = capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame = np.array(yolov4_tiny.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video", frame)
    cv2.waitKey(1)
