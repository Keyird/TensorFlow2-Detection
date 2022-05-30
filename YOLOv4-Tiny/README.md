## 实战介绍与说明
### 1、代码结构说明
```bash
yolov4-tiny
├── data          // 存放预训练模型、类别等数据文件
├── img           // 存放测试图片
├── nets          // 存放各个局部网络结构
├── utlis         // 其他
├── VOCdevkit     // 数据集
├── yolo.py       // 预测过程中的前向推理
├── make_data.py  // 生成标签和图片路径
├── train.py      // 训练网络
├── predict.py    // 对单张图片进行预测
├── video.py      // 对视频进行预测
```
### 2、如何使用本项目进行预测
(1) 如果您希望直接使用本文的模型进行预测，只需完成以下几步：
- 下载网络模型放到data文件夹，下载数据集放在根目录下
- 运行predict.py对图片进行预测。如果是自己的图片，在predict.py中改变图片路径即可。

(2) 如果您需要自建数据集，并对其进行训练和预测，需完成如下步骤：
- 下载预训练模型放到data文件夹下
- 按照VOC2007的格式自制数据集，并放到根目录下
- 新建voc_classes.txt文件，写入类别，并放入data文件夹下
- 运行VOCdevkit下的dataSplit对数据集进行划分
- 运行make_data.py对标签进行解析
- 根据需要更改train.py文件中的anchors_size ( 这一步可选择性跳过，会影响检测效果)
- 运行train.py进行训练，训练完成后，生成的模型默认存放在logs文件下，选择合适的模型最为最终的模型。
- 修改frcnn.py中的model_path，更改为训练好的最终的模型的路径。

### 3、数据集与预训练模型下载链接
在文中获取：[TensorFlow2深度学习实战（十八）：目标检测算法YOLOv4-Tiny实战](https://ai-wx.blog.csdn.net/article/details/124985468)
