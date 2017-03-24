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
        self.conv1 = nn.Conv2d(1, 512, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3200, 640)
        self.fc2 = nn.Linear(640, 128)
        self.fc3 = nn.Linear(128, 81)
        self.steps_done = 0
        self.matches_done = 0
        self.win_count = 0
        self.lose_count = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


env.reset()

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

if os.path.exists('model'):
    model = torch.load('model')
else:
    model = DQN()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())
model.type(dtype)


def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * model.steps_done / EPS_DECAY)
    model.steps_done += 1
    x_min, x_max, y_min, y_max = env.board.move_area()
    if sample > eps_threshold:
        data = model(Variable(state.type(dtype), volatile=True)).data
        k = 1
        m = data.topk(k)[1].min()
        print 'prepare to move on (%d, %d), q = %d' % (m / 9, m % 9, data.topk(k)[0][0][k-1])
        while state[0][0][m / 9][m % 9] != 0 or m / 9 not in range(x_min, x_max) or m % 9 not in range(y_min, y_max):
            k += 1
            m = data.topk(k)[1][0][k-1]
            print 'prepare to move on (%d, %d), q = %f' % (m / 9, m % 9, data.topk(k)[0][0][k - 1])
        return torch.LongTensor([[m]])
    else:
        rand = random.randrange(81)
        print 'random move (%d, %d)' % (rand / 9, rand % 9)
        while state[0][0][rand / 9][rand % 9] != 0 or rand / 9 not in range(x_min, x_max) or rand % 9 not in range(y_min, y_max):
            rand = random.randrange(81)
            print 'random move (%d, %d)' % (rand / 9, rand % 9)
        return torch.LongTensor([[rand]])

last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
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

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    print 'steps_done %d' % model.steps_done
    if model.steps_done % 100 == 0:
        torch.save(model, 'model')
        print 'model saved'


for i_episode in count(1):
    env.reset()
    observation = env.step(40)
    state = observation[0]
    state = torch.from_numpy(state)
    state = state.type(dtype)
    state = state.unsqueeze(0).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, done, _ = env.step(action[0, 0])
        env.render()
        if reward == 10:
            model.matches_done += 1
            model.win_count += 1
            print 'win rate %d / %d' % (model.win_count, model.matches_done)
        elif reward == -10:
            model.matches_done += 1
            model.lose_count += 1
            print 'win rate %d / %d' % (model.win_count, model.matches_done)
        reward = torch.Tensor([reward])
        if not done:
            next_state = observation
            next_state = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0)
            next_state = next_state.type(dtype)
        else:
            next_state = None
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        if done:
            break


