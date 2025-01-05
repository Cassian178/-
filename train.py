import torch
import torchvision
from model import Gender_Detector

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

#用GPU训练
device = torch.device("cuda")

#图像处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
#数据集路径
train_data_dir = "datasets/Training"
test_data_dir = "datasets/Validation"

#加载数据集并以子文件夹名称作为标签
train_data = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_data_dir, transform=transform)

#数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

#加载数据
train_dataloader = DataLoader(train_data,64,True,drop_last=True)
test_dataloader = DataLoader(test_data,64,True,drop_last=True)

gender_detector = Gender_Detector()
gender_detector = gender_detector.to(device)

#损失函数
loss_fn = nn.BCEWithLogitsLoss()
loss_fn = loss_fn.to(device)

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(gender_detector.parameters(),lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 25

writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("----------------第{}轮训练开始---------------------".format(i+1))

    gender_detector.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets_2d = torch.zeros((64, 2), dtype=torch.float32)
        targets_2d.scatter_(1, targets.unsqueeze(1).long(), 1)
        targets_2d = targets_2d.to(device)
        outputs = gender_detector(imgs)
        loss = loss_fn(outputs,targets_2d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if total_train_step % 100 == 0:
            print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)


    gender_detector.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets= targets.to(device)
            targets_2d = torch.zeros((64, 2), dtype=torch.float32).to(device)
            targets_2d.scatter_(1, targets.unsqueeze(1).long(), 1)
            outputs = gender_detector(imgs)
            loss = loss_fn(outputs,targets_2d)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1

    torch.save(gender_detector,"gender_detector_{}.pth".format(i))
    print("模型已保存")

writer.close()



