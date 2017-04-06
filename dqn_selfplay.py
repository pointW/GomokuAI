import os
import random
from board1 import Board

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*Board.BOARD_SIZE_SQ, Board.BOARD_SIZE_SQ)

        self.steps_done = 0
        self.matches_done = 0
        self.win_count = 0
        self.memory = ReplayMemory(10000)

    def forward(self, x):
        moves = []
        for i in range(x.size(0)):
            moves.append(x.data.clone()[i][2].unsqueeze(0))
        moves = torch.cat(moves)
        moves = moves.view(moves.size(0), -1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        neg = moves
        for i in range(neg.size(0)):
            for j in range(neg.size(1)):
                if neg[i][j] == 0:
                    neg[i][j] += 100
                else:
                    neg[i][j] = 0
        neg = Variable(neg)
        x -= neg
        return x


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
model = torch.load('dqn_model')


def select_action(observation):
    state = torch.from_numpy(observation).unsqueeze(0).type(dtype)
    if state[0][0].sum() == 0:
        return 84
    data = model(Variable(state.type(dtype), volatile=True)).data.sort()
    moves = data[1][0]
    q = data[0][0]
    k = 0
    m = moves[Board.BOARD_SIZE_SQ-1 - k]
    print 'opponent prepare to move on (%d, %d), q = %f' % (m / Board.BOARD_SIZE, m % Board.BOARD_SIZE, q[Board.BOARD_SIZE_SQ-1-k])
    while not state[0][2][m / Board.BOARD_SIZE][m % Board.BOARD_SIZE]:
        k += 1
        m = moves[Board.BOARD_SIZE_SQ-1 -k]
        print 'opponent prepare to move on (%d, %d), q = %f' % (m / Board.BOARD_SIZE, m % Board.BOARD_SIZE, q[Board.BOARD_SIZE_SQ-1-k])
    return m


def update_model():
    global model
    model = torch.load('dqn_model')
    model.type(dtype)
