#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/9 12:58
# @Author : ZhangKuo
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        # 3，210，160
        super(DQN, self).__init__()
        self.cov1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        # 32，51，39
        self.cov2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 64，24，18
        self.cov3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 64，22，16
        self.fc1 = nn.Linear(22 * 16 * 64, 512)
        # 512
        self.fc2 = nn.Linear(512, 256)
        # 256
        self.output = nn.Linear(256, output_dim)
        # 4

    def forward(self, x):
        x = torch.relu(self.cov1(x))
        x = torch.relu(self.cov2(x))
        x = torch.relu(self.cov3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        return x
