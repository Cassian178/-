import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#图像处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
#四点裁剪矩形图片并缩放到128*128固定尺寸
def crop_image(image_path, coords_tensor):
    coords = coords_tensor.int().tolist()
    x1, y1, x2, y2 = coords
    with Image.open(image_path) as img:
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_tensor = transform(cropped_img)
        return cropped_tensor
#自定义的绘制函数
def draw_image(image_np, boxes, labels, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale=0.5, font_color=(0, 255, 0), thickness=2):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        # 绘制矩形框
        cv2.rectangle(image_np, (x1, y1), (x2, y2), font_color, thickness)
        # 计算标签的位置
        label_size, baseLine = cv2.getTextSize(label, font_face, font_scale, thickness)
        label_left = x1
        label_top = y1 - label_size[1]
        if label_top < 0:
            label_top += label_size[1]
        # 绘制标签
        cv2.putText(image_np, label, (label_left, label_top), font_face, font_scale, font_color, thickness, cv2.LINE_AA)
    return image_np

img_path = "F:\\OIP-C(7).jpg"
img = cv2.imread(filename = img_path)
# 加载模型
model_yolo = YOLO(model = "runs/detect/train4/weights/best.pt")
model_gender = torch.load("trained_models/gender_detector_23.pth")
# 进行预测
results = model_yolo.predict(source=img,conf=0.2)

# 提取并打印检测结果
for result in results:
    boxes = result.boxes.xyxy  # 边界框坐标
    scores = result.boxes.conf  # 置信度分数
    classes = result.boxes.cls  # 类别索引
    class_names = [result.names[int(cls)] for cls in classes]
    result.names[0] = 'man'
    result.names[1] = 'woman'
    for index, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        print(f"Class: {cls}, Score: {score:.2f}, Box: {box}")
        #参数传递给性别检测网络
        input = crop_image(img_path, box)
        input = input.unsqueeze(0)
        input=input.to(device)
        print(input.shape)
        outputs = model_gender(input)
        #归一化
        probabilities = torch.sigmoid(outputs)
        #输出性别检测结果
        if probabilities[0][0] > 0.5:
            classes_np = classes.cpu().numpy()
            classes_np[index] = 1
            classes = torch.tensor(classes_np, dtype=torch.long)
        else:
            classes_np = classes.cpu().numpy()
            classes_np[index] = 0
            classes = torch.tensor(classes_np, dtype=torch.long)
        #获取类别名称
        class_names = [result.names[int(cls)] for cls in classes]
        # 可视化检测结果
    annotated_img = draw_image(img,boxes,class_names)
    # 在图像上绘制检测结果
    cv2.imshow('Detected Image', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


