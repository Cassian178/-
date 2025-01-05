# -
一个基于yolov8和卷积网络的性别识别项目，用于深度学习的练手

项目的目标是完成现实场景下的人脸识别定位操作，并且根据识别出的人脸进行目标的性别识别。人脸目标检测使用经人脸训练集从头训练了250轮的yolov8模型，性别二分类模型使用自己搭建的卷积网络模型。检测结果的可视化以及摄像头调用通过OpenCV库实现。

以下是文件功能详述:

datasets：性别数据集

datasets_for_yolo：人脸数据集

runs：保存yolo训练数据和pt文件

trained_models:  保存二分类卷积网络pt文件，23为best,24为last

camera_detect.py: 电脑摄像头检测

picture_detect.py: 图片检测

test.py: 测试和调试一些代码好不好使的，没啥意义

train.py :卷积网络训练文件

train_yolov8.py:训练yolov8
