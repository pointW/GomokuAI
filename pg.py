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
import numpy as np

from torch.autograd import Variable

env = Env()

Transition = namedtuple('Transition', ('state', 'action', 'y', 'reward'))
memory = []


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


if os.path.exists('pg_model'):
    model = torch.load('pg_model')
else:
    model = PolicyNetwork()

# optimizer = optim.RMSprop(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.128)
model.type(dtype)


def discount_rewards(r):
    discounted_r = np.zeros(r.size()[0])
    running_add = 0
    for t in reversed(xrange(0, r.size()[0])):
        if r[t][0] != 0: running_add = 0
        running_add = running_add * GAMMA + r[t][0]
        discounted_r[t] = running_add
    discounted_r = torch.from_numpy(discounted_r)
    discounted_r = discounted_r.type(dtype)
    return discounted_r


def select_action(state):
    if state[0][0].sum() == 0:
        return 40
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * model.steps_done / EPS_DECAY)
    model.steps_done += 1
    if sample > eps_threshold:
        data = model(Variable(state.type(dtype), volatile=True)).data.sort()
        moves = data[1][0]
        probs = data[0][0]
        k = 0
        m = moves[80-k]
        print 'prepare to move on (%d, %d), p = %f' % (m / 9, m % 9, probs[80-k])
        while not state[0][2][m / 9][m % 9]:
        # while state[0][0][m / 9][m % 9] or state[0][1][m / 9][m % 9]:
            k += 1
            m = moves[80-k]
            print 'prepare to move on (%d, %d), p = %f' % (m / 9, m % 9, probs[80-k])
        return m
    else:
        rand = random.randrange(81)
        print 'random move (%d, %d)' % (rand / 9, rand % 9)
        while not state[0][2][rand / 9][rand % 9]:
        # while state[0][0][rand / 9][rand % 9] or state[0][1][rand / 9][rand % 9]:
            rand = random.randrange(81)
            print 'random move (%d, %d)' % (rand / 9, rand % 9)
        return rand


def optimize_model():
    # global states, rewards, ys, actions
    # epstate = torch.cat(states)
    # epy = torch.cat(ys)
    # epreward = torch.cat(rewards)
    # epaction = torch.cat(actions)
    #
    # states, rewards, ys, actions = [], [], [], []

    global memory
    batch = Transition(*zip(*memory))
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = torch.cat(batch.reward)
    memory = []
    reward_batch = discount_rewards(reward_batch)
    reward_batch = Variable(reward_batch)
    epprob = model(state_batch).gather(1, action_batch).cpu()
    loglik = torch.log(epprob)
    loss = -torch.mean(loglik*reward_batch)


    # output = model(Variable(epstate.type(dtype)))
    # discounted_epr = discount_rewards(epreward)
    # discounted_epr.resize_(discounted_epr.size()[0], 1)
    # epaction = Variable(epaction)
    # epprob = output.gather(1, epaction)
    # epy = torch.ones(discounted_epr.size())
    # for i in range(discounted_epr.size()[0]):
    #     if discounted_epr[i][0] <= 0:
    #         epy[i][0] = 0
    # discounted_epr = Variable(discounted_epr)
    # epy = Variable(epy)
    # loglik = torch.log(epprob)
    # loss = F.smooth_l1_loss(loglik*discounted_epr, torch.log(epy))

    # output = model(Variable(epstate.type(dtype)))
    # discounted_epr = discount_rewards(epreward)
    # discounted_epr.resize_(discounted_epr.size()[0], 1)
    # epaction = Variable(epaction)
    # epprob = output.gather(1, epaction)
    # epy = torch.zeros(discounted_epr.size())
    # for i in range(discounted_epr.size()[0]):
    #     if discounted_epr[i][0] <= 0:
    #         epy[i][0] = 1
    #         discounted_epr[i][0] = -discounted_epr[i][0]
    # discounted_epr = Variable(discounted_epr)
    # epy = Variable(epy)
    # loglik = -torch.log(epy*(epy-epprob)+(1-epy)*(epy+epprob))
    # loss = (loglik*discounted_epr).mean()

    # discounted_epr = discounted_epr.expand(discounted_epr.size()[0], 81)
    #
    # epy = Variable(epy)
    # discounted_epr = Variable(discounted_epr)
    #
    # loglik = torch.log(epy*(epy-output)+(1-epy)*(epy+output))
    #
    # loss = (loglik*discounted_epr).mean()

    f = open('pg_loss', 'a')
    print >> f, "loss = %f, match = %d, step = %d" % (loss.data[0], model.matches_done, model.steps_done)
    f.close()

    optimizer.zero_grad()
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

    print 'steps_done %d' % model.steps_done
    torch.save(model, 'pg_model')
    print 'model saved'

env.reset()
observation = env.make_observation_7()
while True:
    state = torch.from_numpy(observation).unsqueeze(0).type(dtype)
    action = select_action(state)
    observation, reward, done, _ = env.step_pg(action)
    # if reward == -10:
    #     if rewards[-1][0][0] == -5:
    #         reward = -2

    env.render()
    print 'reward = %.1f' % reward
    model.steps_done += 1

    # states.append(state)
    # rewards.append(torch.FloatTensor([[reward]]))
    # actions.append(torch.LongTensor([[action]]))
    y = (torch.zeros(81)).unsqueeze(0)
    y[0][action] = 1
    # ys.append(y)

    memory.append(Transition(state, torch.LongTensor([[action]]), y, torch.FloatTensor([[reward]])))

    state = torch.from_numpy(observation).unsqueeze(0)

    if done:
        model.matches_done += 1
        if reward == 10:
            model.win_count += 1
        print 'win rate %d / %d' % (model.win_count, model.matches_done)
        optimize_model()
        env.reset()
        observation = env.make_observation_7()
