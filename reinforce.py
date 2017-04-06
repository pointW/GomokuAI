import os
import random
import math
from collections import namedtuple
from itertools import count
from env import Env

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from torch.autograd import Variable

env = Env()


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(7, 128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(1)

        self.steps_done = 0
        self.matches_done = 0
        self.win_count = 0

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = x.view(x.size(0), -1)
        x = F.softmax(x)
        return x

GAMMA = 0.9
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 2000
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
states, rewards, ys, actions = [], [], [], []

if os.path.exists('reinforce_model'):
    model = torch.load('reinforce_model')
else:
    model = PolicyNetwork()

optimizer = optim.SGD(model.parameters(), lr=0.128)
model.type(dtype)