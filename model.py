import torch
import torch.nn as nn
import torch.nn.functional

#CNN网络
class Gender_Detector(nn.Module):
    def __init__(self):
        super(Gender_Detector, self).__init__()
        self.model = nn.Sequential(
            #第一层卷积
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #第二春卷积
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #第三层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #展平
            nn.Flatten(),

            #原图尺寸统一为128*128，通道64
            nn.Linear(64*16*16, 256),
            nn.ReLU(),
            #减少过拟合
            nn.Dropout(0.2),

            nn.Linear(256, 2)
        )


    def forward(self, x):
        x=self.model(x)
        return x
#测试模型正确性
if __name__ == '__main__':
    model_test = Gender_Detector()
    input = torch.ones((1, 3, 128, 128))
    output = model_test(input)
    print(output.shape)


