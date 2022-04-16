"""
数据集按照如下比例划分：
训练集+验证集：测试集 = 9：1
训练集：验证集 = 9：1
"""
import os
import random 
random.seed(0)

xmlfilepath = "./Annotations"
saveBasePath = "./ImageSets/Main/"

temp_xml = os.listdir(xmlfilepath)  # 获取xmlfilepath路径下所有的文件名
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):  # 对于后缀为.xml的文件名存入数组total_xml中
        total_xml.append(xml)

num = len(total_xml)  # 样本的总数
trainval_percent = 0.9
train_percent = 0.9

trainval_length = int(num * trainval_percent)
train_length = int(trainval_length * train_percent)

list = range(num)  # [0, num)的列表
trainval = random.sample(list, trainval_length)
train = random.sample(trainval, train_length)

ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
 
for i in list:
    name = total_xml[i][:-4]+'\n'
    if i in trainval:
        if i in train:  
            ftrain.write(name)  # 训练集
        else:  
            fval.write(name)    # 验证集
    else:  
        ftest.write(name)       # 测试集

ftrain.close()  
fval.close()  
ftest .close()
