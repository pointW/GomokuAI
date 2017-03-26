import gym
import gym_gomoku
from board1 import Board
import numpy as np


class Env2(object):
    def __init__(self):
        self.env = gym.make('Gomoku9x9-v0')  # default 'beginner' level opponent policy
        self.board = Board()
        self.reward = 0

    def reset(self):
        self.env.reset()
        self.board = Board()
        self.reward = 0

    def step(self, mv):
        observation, reward, _, info = self.env.step(mv)
        pre_state = self.board.stones
        observation = observation.reshape(81)
        diff = np.where((pre_state != observation))[0]
        if observation[diff[0]] == self.board.STONE_BLACK:
            move1 = diff[0]
            move2 = diff[1]
        else:
            move1 = diff[1]
            move2 = diff[0]
        self.board.move(move1, self.board.STONE_BLACK)
        done, _ = self.board.is_over()
        if done:
            observation = self.board.stones.reshape(9, 9)
            self.reward = 1
            _ = None
            return observation, self.reward, done, _
        else:
            self.reward = self.get_reward()
            self.board.move(move2, self.board.STONE_WHITE)
            done, _ = self.board.is_over()
            observation = self.board.stones.reshape(9, 9)
            _ = None
            if done:
                self.reward = -1
            else:
                new_reward = self.get_reward()
                self.reward += new_reward
            return observation, self.reward, done, _

    def get_reward(self):
        value = self.board.find_pattern()
        reward = value*0.1
        if self.board.stones[self.board.last_move] == self.board.STONE_BLACK:
            return reward
        else:
            return self.reward - reward


env = Env2()
observation, reward, done, _ = env.step(40)
a = 1

