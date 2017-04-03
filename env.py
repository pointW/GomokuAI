import gym
import gym_gomoku
from board1 import Board
from opponent import Searcher
import numpy as np
import random


class Env(object):
    reward = 0.0
    BLACK = Board.STONE_BLACK
    WHITE = Board.STONE_WHITE
    EMPTY = Board.STONE_EMPTY
    REWARD_WIN = 10
    REWARD_LOSE = -10

    def __init__(self):
        self.board = Board()
        self.opponent = Searcher()
        self.env = gym.make('Gomoku9x9-v0')  # default 'beginner' level opponent policy
        self.env.reset()
        self.type = 1

    def reset(self):
        self.board = Board()
        self.opponent = Searcher()
        self.env.reset()
        self.type = 1

    def step(self, mv):
        if self.type == 1:
            return self.step_1(mv)
        elif self.type == 2:
            return self.step_2(mv)

    def step_pg(self, mv):
        self.board.move(mv, self.board.STONE_BLACK)
        done, _ = self.board.is_over()
        if done:
            observation = self.make_observation_7()
            self.reward = self.REWARD_WIN
            _ = None
            return observation, self.reward, done, _
        else:
            self.reward = self.get_reward()
            self.opponent.board = self.board.stones.reshape(9, 9)
            _, row, col = self.opponent.search(self.board.STONE_WHITE, depth=1)
            self.board.move(9 * row + col, self.board.STONE_WHITE)
            done, _ = self.board.is_over()
            observation = self.make_observation_7()
            _ = None
            if done:
                self.reward = self.REWARD_LOSE
            else:
                self.reward = self.get_reward()
            return observation, self.reward, done, _

    def step_1(self, mv):
        self.board.move(mv, self.board.STONE_BLACK)
        done, _ = self.board.is_over()
        if done:
            observation = self.make_observation()
            self.reward = self.REWARD_WIN
            _ = None
            return observation, self.reward, done, _
        else:
            self.reward = self.get_reward()
            self.opponent.board = self.board.stones.reshape(9, 9)
            _, row, col = self.opponent.search(self.board.STONE_WHITE, depth=1)
            self.board.move(9 * row + col, self.board.STONE_WHITE)
            done, _ = self.board.is_over()
            observation = self.make_observation()
            _ = None
            if done:
                self.reward = self.REWARD_LOSE
            else:
                self.reward = self.get_reward()
            return observation, self.reward, done, _

    def step_2(self, mv):
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
            observation = self.make_observation()
            self.reward = self.REWARD_WIN
            _ = None
            return observation, self.reward, done, _
        else:
            self.reward = self.get_reward()
            self.board.move(move2, self.board.STONE_WHITE)
            done, _ = self.board.is_over()
            observation = self.make_observation()
            _ = None
            if done:
                self.reward = self.REWARD_LOSE
            else:
                self.reward = self.get_reward()
            return observation, self.reward, done, _

    def make_observation(self):
        observation_black = []
        observation_white = []
        for i in range(81):
            if self.board.stones[i] == self.board.STONE_BLACK:
                observation_black.append(1)
                observation_white.append(0)
            elif self.board.stones[i] == self.board.STONE_WHITE:
                observation_white.append(1)
                observation_black.append(0)
            else:
                observation_black.append(0)
                observation_white.append(0)
        observation_black = np.array(observation_black)
        observation_white = np.array(observation_white)
        observation = np.array([observation_black.reshape(9, 9), observation_white.reshape(9, 9)])
        return observation

    def make_observation_7(self):
        observation_black = []
        observation_white = []
        # observation_empty = []
        observation_empty = np.zeros(81).reshape(9, 9)
        for i in range(81):
            if self.board.stones[i] == self.board.STONE_BLACK:
                observation_black.append(1)
                observation_white.append(0)
                # observation_empty.append(0)
            elif self.board.stones[i] == self.board.STONE_WHITE:
                observation_white.append(1)
                observation_black.append(0)
                # observation_empty.append(0)
            else:
                # observation_empty.append(1)
                observation_black.append(0)
                observation_white.append(0)
        x_min, x_max, y_min, y_max = self.board.move_area()
        for i in range(x_min, x_max+1):
            for j in range(y_min, y_max+1):
                if observation_black[i*9+j] == 0 and observation_white[i*9+j] == 0:
                    observation_empty[i][j] = 1
        observation_black = np.array(observation_black).reshape(9, 9)
        observation_white = np.array(observation_white).reshape(9, 9)
        observation_empty = np.array(observation_empty).reshape(9, 9)
        observation_atk_1, observation_atk_2 = self.board.gen_dfs_atk_moves(1)
        observation_dfs_1, observation_dfs_2 = self.board.gen_dfs_atk_moves(0)
        observation = np.array([observation_black, observation_white, observation_empty,
                                observation_atk_1, observation_atk_2,
                                observation_dfs_1, observation_dfs_2])
        return observation

    def render(self):
        b = self.board.stones.reshape(9, 9)
        p = '  0 1 2 3 4 5 6 7 8 \n'
        for i in range(9):
            p += str(i) + ' '
            for j in range(9):
                if b[i][j] == self.board.STONE_BLACK:
                    p += 'X '
                elif b[i][j] == self.board.STONE_WHITE:
                    p += 'O '
                else:
                    p += '. '
            p += '\n'
        print p
        print '%d moves in %d, %d' % (self.board.stones[self.board.last_move], self.board.last_move / 9, self.board.last_move % 9)

    def get_reward(self):
        # c, p = self.board.find_pattern()
        # if c == self.board.STONE_BLACK and (p == 3 or p == 4):
        #     return 1
        # if c == self.board.STONE_WHITE and (p == 3 or p == 4):
        #     return -5
        # return 0
        value = self.board.find_pattern()
        reward = value
        if self.board.stones[self.board.last_move] == self.board.STONE_BLACK:
            return reward
        else:
            return self.reward - reward

