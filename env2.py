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
        move1 = diff[0]
        move2 = diff[1]
        self.board.move(move1, self.board.STONE_BLACK)
        done, _ = self.board.is_over()
        if done:
            observation = self.board.stones.reshape(9, 9)
            self.reward = 10
            _ = None
            return observation, self.reward, done, _
        else:
            self.reward = self.get_reward()
            self.board.move(move2, self.board.STONE_WHITE)
            done, _ = self.board.is_over()
            observation = self.board.stones.reshape(9, 9)
            _ = None
            if done:
                self.reward = -10
            else:
                new_reward = self.get_reward()
                self.reward += new_reward
            return observation, self.reward, done, _

    def get_reward(self):
        c, p = self.board.find_pattern()
        if c == self.board.STONE_BLACK and (p == 3 or p == 4):
            return 1
        if c == self.board.STONE_WHITE and (p == 3 or p == 4):
            return -5
        return 0


env = Env2()
env.step(40)
