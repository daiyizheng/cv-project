# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2022/1/13 14:50
# software: PyCharm

"""
文件说明：
    
"""
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        ## height = (height_in - height_kernel + 2*padding)/stride + 1
        ## width = (width_in - width_kernel + 2*padding)/stride + 1
        ## 第一层卷积层，kernel_size为3*3，out_channel为96，stride为4，初始图像大小为227*227, padding为2
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 * 3, stride=2, padding=0),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=args.num_class)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x