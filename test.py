# import torch
# import torchvision
#
#
# from torch import nn
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
#
# from torchvision import datasets, transforms
#
#
#
# # # 假设x是一个形状为[N]的向量，只包含0和1
# # N = 5
# # x = torch.randint(0, 2, (N,))  # 生成一个随机包含0和1的[N]向量
# # print(x)
# # # # 创建一个形状为[N, 2]的张量，初始化为0
# # # # 这里我们使用float32类型，因为通常用于神经网络
# # result = torch.zeros((N, 2), dtype=torch.float32)
# # #
# # # # 使用条件索引来设置result的值
# # # # 对于x中的每个元素，如果它是0，则result对应行的[0]位置设为1；如果它是1，则result对应行的[1]位置设为1
# # # # 注意，这里我们利用了x的布尔值来索引result，这在PyTorch中是有效的
# # result.scatter_(1, x.unsqueeze(1).long(), 1)
# # #
# # # # 检查结果
# # print(result)
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])
# #数据集路径
# train_data_dir = "datasets/Training"
# test_data_dir = "datasets/Validation"
# train_data = datasets.ImageFolder(root=train_data_dir, transform=transform)
# test_data = datasets.ImageFolder(root=test_data_dir, transform=transform)
#
# train_dataloader = DataLoader(train_data,64,True)
# test_dataloader = DataLoader(test_data,64,True)
# for data in train_dataloader:
#     imgs, targets = data
#
#     targets_2d = torch.zeros((64, 2), dtype=torch.float32)
#     targets_2d.scatter_(1, targets.unsqueeze(1).long(),1)
#     print(targets)
#     print(targets_2d)

# import cv2
#
# from ultralytics import YOLO
#
# img_path = "F:\\f9bfa29d765863dc2980f3c1a98619da.jpeg"
# img = cv2.imread(filename = img_path)
# model = YOLO(model = "../runs/detect/train11/weights/best.pt")
# res = model(img)
# annotated_img = res[0].plot()
# cv2.imshow(winname="yolo",mat = annotated_img)
# cv2.waitKey(delay=10000)

import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class_names = {'female','male'}
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
def crop_image_with_tensor_coords(image_path, coords_tensor):
    # 假设 coords_tensor 是 [x1, y1, x2, y2] 的张量，我们需要将其转换为整数坐标
    # 这里我们使用 floor 函数来向下取整坐标，但你也可以选择 round 或其他方式
    coords = coords_tensor.int().tolist()
    x1, y1, x2, y2 = coords

    # 使用 PIL 打开图像
    with Image.open(image_path) as img:
        # PIL 的 crop 方法需要左上角和右下角的坐标，格式为 (left, top, right, bottom)
        cropped_img = img.crop((x1, y1, x2, y2))

        cropped_tensor = transform(cropped_img)

        return cropped_tensor


def draw_rectangles_and_labels_on_numpy_image(image_np, boxes, labels, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale=0.5, font_color=(0, 255, 0), thickness=2):
    # 确保image_np是HWC格式的RGB图像
    # 注意：OpenCV期望BGR格式，但在这里我们直接在RGB上绘制（颜色顺序需正确）

    # 遍历每个矩形框和对应的标签
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)

        # 绘制矩形框
        cv2.rectangle(image_np, (x1, y1), (x2, y2), font_color, thickness)

        # 计算标签的位置（这里简单地放在矩形框的左上角）
        # 注意：你可能需要根据标签的长度和字体大小来调整这个位置
        label_size, baseLine = cv2.getTextSize(label, font_face, font_scale, thickness)
        label_left = x1
        label_top = y1 - label_size[1]

        # 如果标签超出了图像边界，你可能需要调整位置
        if label_top < 0:
            label_top += label_size[1]  # 简单地将其移到矩形框的下方

        # 绘制标签
        cv2.putText(image_np, label, (label_left, label_top), font_face, font_scale, font_color, thickness, cv2.LINE_AA)

        # 注意：我们没有进行颜色空间转换，因为在这个例子中它不影响绘制矩形和文本
    # 但如果你需要与其他期望BGR格式的OpenCV函数一起使用，请记得转换

    return image_np
def draw_rectangles_and_labels(image, coordinates, labels):
    """
    在PIL图像上绘制矩形框和标签。

    参数:
    - image: PIL Image对象
    - coordinates: 列表的列表或列表的元组，形状为 [N, 4]，每个子列表/元组包含四个元素 (x1, y1, x2, y2)
    - labels: 列表的字符串，每个字符串对应一个矩形框的标签

    返回:
    - PIL Image: 带有矩形框和标签的图像
    """
    # 绘制矩形框和标签
    draw = ImageDraw.Draw(image)
    # 假设使用默认字体，你可能需要安装字体文件或使用系统字体
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for coord, label in zip(coordinates, labels):
        x1, y1, x2, y2 = coord
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # 绘制标签，这里假设标签在矩形框左上角旁边的位置
        text_width, text_height = draw.textsize(label, font)
        text_pos = (x1, y1 - text_height - 5)  # 根据需要调整位置
        # 如果标签位置超出图像边界，则进行适当调整（这里仅作为示例，未完全处理所有边界情况）
        if text_pos[1] < 0:
            text_pos = (x1, y1 + 10)  # 将标签移到矩形框下方
        draw.text(text_pos, label, fill="white", font=font)

    return image

img_path = "F:\\OIP-C (3).jpg"
img = cv2.imread(filename = img_path)
image = Image.open(img_path)
model = YOLO(model = "../runs/detect/train11/weights/best.pt")

results = model(img)
print(f"results={results}")

# for i,result in enumerate(results):
#     boxes = result.boxes.xyxy
#     scores = result.boxes.conf
#     classes = result.boxes.cls
#     print(f"boxes={boxes}")
#     print(f"result.names={result.names}")
#     result.names = 'man'
#
#     print(scores)
#     print(result)

for result in results:
    boxes = result.boxes.xyxy  # 边界框坐标
    scores = result.boxes.conf  # 置信度分数
    classes = result.boxes.cls  # 类别索引
    class_names = [model.names[int(cls)] for cls in classes]  # 获取类别名称
    print(f"Class: {classes.tolist()}, Score: {scores.shape}, Box: {boxes.shape}")
    for box, score, class_name in zip(boxes, scores, class_names):
        tensor = crop_image_with_tensor_coords(img_path, box)

        tensor_np = tensor.permute(1, 2, 0).numpy()  # 从[C, H, W]到[H, W, C]

        tensor_np = (tensor_np * 255).astype(np.uint8)
        img = Image.fromarray(tensor_np, 'RGB')
        img.show()
        print(f"Class: {class_name}, Score: {score:.2f}, Box: {box}")

#     # 可视化检测结果
    annotated_img = draw_rectangles_and_labels_on_numpy_image(img, boxes, class_names)

    cv2.imshow('Detected Image', annotated_img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
# 假设results是模型推理后返回的检测结果列表
# for i, det in enumerate(results):
#     # 假设我们要调整第一个检测到的对象的边界框
#     if i == 0:
#         det['x1'] += 10  # 向右移动边界框
#         det['y1'] += 10  # 向下移动边界框

# annotated_img = results[0].plot()  # 在图像上绘制检测结果
