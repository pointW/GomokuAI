import gym
import gym_gomoku
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

from torch.autograd import Variable

# env = gym.make('Gomoku9x9-v0') # default 'beginner' level opponent policy
env = Env()

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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
        self.conv1 = nn.Conv2d(7, 64, kernel_size=5, stride=1, padding=2)
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
        self.fc = nn.Linear(128*9*9, 81)

        # self.fc1 = nn.Linear(64*9*9, 1024)
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(256, 81)

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
        # x = self.fc1(x.view(x.size(0), -1))
        # x = self.fc2(x)
        # x = self.fc3(x)
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


env.reset()

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

if os.path.exists('dqn_model'):
    model = torch.load('dqn_model')
else:
    model = DQN()

# optimizer = optim.RMSprop(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.128)
model.type(dtype)


def select_action(state):
    model.steps_done += 1
    if state[0][0].sum() == 0:
        return 40
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * model.steps_done / EPS_DECAY)
    if sample > eps_threshold:
        data = model(Variable(state.type(dtype), volatile=True)).data.sort()
        moves = data[1][0]
        q = data[0][0]
        k = 0
        m = moves[80 - k]
        print 'prepare to move on (%d, %d), q = %f' % (m / 9, m % 9, q[80-k])
        while not state[0][2][m / 9][m % 9]:
            k += 1
            m = moves[80-k]
            print 'prepare to move on (%d, %d), q = %f' % (m / 9, m % 9, q[80-k])
        return m
    else:
        rand = random.randrange(81)
        print 'random move (%d, %d)' % (rand / 9, rand % 9)
        while not state[0][2][rand / 9][rand % 9]:
            rand = random.randrange(81)
            print 'random move (%d, %d)' % (rand / 9, rand % 9)
        return rand


def optimize_model():
    if len(model.memory) < BATCH_SIZE:
        return
    transitions = model.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t, volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    if USE_CUDA:
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()

    state_action_values = model(state_batch).gather(1, action_batch).cpu()

    next_state_values = Variable(torch.zeros(BATCH_SIZE))

    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].cpu()

    next_state_values.volatile = False

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    f = open('dqn_loss', 'a')
    print >> f, "loss = %f, match = %d, step = %d" % (loss.data[0], model.matches_done, model.steps_done)
    f.close()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print 'steps_done %d' % model.steps_done
    if model.steps_done % 100 == 0:
        torch.save(model, 'dqn_model')
        print 'model saved'

env.reset()
observation = env.make_observation_7()
while True:
    state = torch.from_numpy(observation).unsqueeze(0).type(dtype)
    action = select_action(state)
    observation, reward, done, _ = env.step_pg(action)
    env.render()
    print 'reward = %.1f' % reward
    if not done:
        next_state = observation
        next_state = torch.from_numpy(next_state).unsqueeze(0)
        next_state = next_state.type(dtype)
    else:
        next_state = None
        model.matches_done += 1
        if reward == env.REWARD_WIN:
            model.win_count += 1
        print 'win rate %d / %d' % (model.win_count, model.matches_done)
        env.reset()
        observation = env.make_observation_7()
    reward = torch.FloatTensor([reward])
    model.memory.push(state, torch.LongTensor([[action]]), next_state, reward)
    optimize_model()

