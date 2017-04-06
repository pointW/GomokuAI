import gym
import gym_gomoku
from board1 import Board
from opponent import Searcher
import numpy as np
import random
import dqn_selfplay


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
            self.opponent.board = self.board.stones.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
            _, row, col = self.opponent.search(self.board.STONE_WHITE, depth=1)
            self.board.move(Board.BOARD_SIZE * row + col, self.board.STONE_WHITE)
            done, _ = self.board.is_over()
            observation = self.make_observation_7()
            _ = None
            if done:
                self.reward = self.REWARD_LOSE
            else:
                self.reward = self.get_reward()
            return observation, self.reward, done, _

    def step_o9(self, mv):
        self.board.move(mv, self.board.STONE_BLACK)
        done, _ = self.board.is_over()
        if done:
            observation = self.make_observation_9()
            self.reward = self.REWARD_WIN
            _ = None
            return observation, self.reward, done, _
        else:
            self.reward = self.get_reward()
            self.oppo_move()
            done, _ = self.board.is_over()
            observation = self.make_observation_9()
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
            self.opponent.board = self.board.stones.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
            _, row, col = self.opponent.search(self.board.STONE_WHITE, depth=1)
            self.board.move(Board.BOARD_SIZE * row + col, self.board.STONE_WHITE)
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
        observation = observation.reshape(Board.BOARD_SIZE_SQ)
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
        for i in range(Board.BOARD_SIZE_SQ):
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
        observation = np.array([observation_black.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE), observation_white.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)])
        return observation

    def make_observation_7(self):
        observation_black = []
        observation_white = []
        # observation_empty = []
        observation_empty = np.zeros(Board.BOARD_SIZE_SQ).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        for i in range(Board.BOARD_SIZE_SQ):
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
                if observation_black[i*Board.BOARD_SIZE+j] == 0 and observation_white[i*Board.BOARD_SIZE+j] == 0:
                    observation_empty[i][j] = 1
        observation_black = np.array(observation_black).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_white = np.array(observation_white).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_empty = np.array(observation_empty).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_atk_1, observation_atk_2 = self.board.gen_dfs_atk_moves(1)
        observation_dfs_1, observation_dfs_2 = self.board.gen_dfs_atk_moves(0)
        observation = np.array([observation_black, observation_white, observation_empty,
                                observation_atk_1, observation_atk_2,
                                observation_dfs_1, observation_dfs_2])
        return observation

    def make_observation_9(self):
        observation_black = []
        observation_white = []
        observation_empty = np.zeros(Board.BOARD_SIZE_SQ).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        for i in range(Board.BOARD_SIZE_SQ):
            if self.board.stones[i] == self.board.STONE_BLACK:
                observation_black.append(1)
                observation_white.append(0)
            elif self.board.stones[i] == self.board.STONE_WHITE:
                observation_white.append(1)
                observation_black.append(0)
            else:
                observation_black.append(0)
                observation_white.append(0)
        x_min, x_max, y_min, y_max = self.board.move_area()
        for i in range(x_min, x_max+1):
            for j in range(y_min, y_max+1):
                if observation_black[i*Board.BOARD_SIZE+j] == 0 and observation_white[i*Board.BOARD_SIZE+j] == 0:
                    observation_empty[i][j] = 1
        observation_black = np.array(observation_black).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_white = np.array(observation_white).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_empty = np.array(observation_empty).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_atk_1, observation_atk_2, observation_atk_3, observation_atk_4 = self.board.gen_dfs_atk_moves(1)
        observation_dfs_1, observation_dfs_2 = self.board.gen_dfs_atk_moves(0)
        observation = np.array([observation_black, observation_white, observation_empty,
                                observation_atk_1, observation_atk_2, observation_atk_3, observation_atk_4,
                                observation_dfs_1, observation_dfs_2])
        return observation

    def make_observation_9_by_v(self, v):
        board = Board()
        board.stones = self.board.stones
        if v == Board.STONE_WHITE:
            board.reverse()
        observation_self = []
        observation_oppo = []
        observation_move = np.zeros(Board.BOARD_SIZE_SQ).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        for i in range(Board.BOARD_SIZE_SQ):
            if board.stones[i] == v:
                observation_self.append(1)
                observation_oppo.append(0)
            elif self.board.stones[i] != 0:
                observation_oppo.append(1)
                observation_self.append(0)
            else:
                observation_self.append(0)
                observation_oppo.append(0)
        x_min, x_max, y_min, y_max = board.move_area()
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                if observation_self[i * Board.BOARD_SIZE + j] == 0 and observation_oppo[i * Board.BOARD_SIZE + j] == 0:
                    observation_move[i][j] = 1
        observation_self = np.array(observation_self).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_oppo = np.array(observation_oppo).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_move = np.array(observation_move).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        observation_atk_1, observation_atk_2, observation_atk_3, observation_atk_4 = board.gen_dfs_atk_moves(1)
        observation_dfs_1, observation_dfs_2 = board.gen_dfs_atk_moves(0)
        observation = np.array([observation_self, observation_oppo, observation_move,
                                observation_atk_1, observation_atk_2, observation_atk_3, observation_atk_4,
                                observation_dfs_1, observation_dfs_2])
        return observation

    def render(self):
        b = self.board.stones.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        p = '  '
        for i in range(Board.BOARD_SIZE):
            p += str(i)
            if i < 10:
                p += ' '
        p += '\n'
        for i in range(Board.BOARD_SIZE):
            p += str(i)
            if i < 10:
                p += ' '
            for j in range(Board.BOARD_SIZE):
                if b[i][j] == self.board.STONE_BLACK:
                    p += 'X '
                elif b[i][j] == self.board.STONE_WHITE:
                    p += 'O '
                else:
                    p += '. '
            p += '\n'
        print p
        print '%d moves in %d, %d' % (self.board.stones[self.board.last_move], self.board.last_move / Board.BOARD_SIZE, self.board.last_move % Board.BOARD_SIZE)

    def get_reward(self):
        # c, p = self.board.find_pattern()
        # if c == self.board.STONE_BLACK and (p == 3 or p == 4):
        #     return 1
        # if c == self.board.STONE_WHITE and (p == 3 or p == 4):
        #     return -5
        # return 0
        value = self.board.find_pattern()
        reward = 0
        if value == 3:
            reward = 1
        elif value == 4:
            reward = 2
        elif value == 5:
            reward = 5
        if self.board.stones[self.board.last_move] == self.board.STONE_BLACK:
            return reward
        else:
            return self.reward - reward

    def quick_play(self):
        observation_atk_1, observation_atk_2, observation_atk_3, observation_atk_4 = self.board.gen_dfs_atk_moves(1)
        observation_dfs_1, observation_dfs_2 = self.board.gen_dfs_atk_moves(0)
        observation_atk_1 = observation_atk_1.reshape(Board.BOARD_SIZE_SQ)
        observation_atk_2 = observation_atk_2.reshape(Board.BOARD_SIZE_SQ)
        observation_atk_3 = observation_atk_3.reshape(Board.BOARD_SIZE_SQ)
        observation_atk_4 = observation_atk_4.reshape(Board.BOARD_SIZE_SQ)
        observation_dfs_1 = observation_dfs_1.reshape(Board.BOARD_SIZE_SQ)
        observation_dfs_2 = observation_dfs_2.reshape(Board.BOARD_SIZE_SQ)
        if observation_atk_4.any():
            print 'atk4'
            return observation_atk_4.argmax()
        elif observation_dfs_2.any():
            pool = np.where(observation_dfs_2 == 1)[0]
            print 'dfs2 pool'
            print pool
            return np.random.choice(pool)
        elif observation_atk_3.any():
            pool = np.where(observation_atk_3 == 1)[0]
            print 'atk3 pool'
            print pool
            return np.random.choice(pool)
        elif observation_dfs_1.any():
            if observation_atk_2.any():
                if random.random() > 0.7:
                    pool = np.where(observation_atk_2 == 1)[0]
                    print 'atk2 pool'
                    print pool
                    return np.random.choice(pool)
                else:
                    pool = np.where(observation_dfs_1 == 1)[0]
                    print 'dfs1 pool'
                    print pool
                    return np.random.choice(pool)
            else:
                pool = np.where(observation_dfs_1 == 1)[0]
                print 'dfs1 pool'
                print pool
                return np.random.choice(pool)
        elif observation_atk_1.any() or observation_atk_2.any():
            pool = np.hstack((np.where(observation_atk_1 == 1)[0], np.where(observation_atk_2 == 1)[0]))
            print 'atk1 atk2 pool'
            print pool
            return np.random.choice(pool)
        else:
            move_area = self.make_observation_9()[2]
            move_area = move_area.reshape(Board.BOARD_SIZE_SQ)
            pool = np.where(move_area == 1)[0]
            print 'random'
            return np.random.choice(pool)

    def oppo_quick_play(self):
        oppo_board = Board()
        oppo_board.stones = self.board.stones.copy()
        oppo_board.reverse()
        atk1, atk2, atk3, atk4 = oppo_board.gen_dfs_atk_moves(1)
        dfs1, dfs2 = oppo_board.gen_dfs_atk_moves(0)
        atk1 = atk1.reshape(Board.BOARD_SIZE_SQ)
        atk2 = atk2.reshape(Board.BOARD_SIZE_SQ)
        atk3 = atk3.reshape(Board.BOARD_SIZE_SQ)
        atk4 = atk4.reshape(Board.BOARD_SIZE_SQ)
        dfs1 = dfs1.reshape(Board.BOARD_SIZE_SQ)
        dfs2 = dfs2.reshape(Board.BOARD_SIZE_SQ)
        if atk4.any():
            return atk4.argmax()
        elif dfs2.any():
            pool = np.where(dfs2 == 1)[0]
            return np.random.choice(pool)
        elif atk3.any():
            pool = np.where(atk3 == 1)[0]
            return np.random.choice(pool)
        elif dfs1.any():
            if atk2.any():
                if random.random() > 0.7:
                    pool = np.where(atk2 == 1)[0]
                    return np.random.choice(pool)
                else:
                    pool = np.where(dfs1 == 1)[0]
                    return np.random.choice(pool)
            else:
                pool = np.where(dfs1 == 1)[0]
                return np.random.choice(pool)
        elif atk1.any() or atk2.any():
            pool = np.hstack((np.where(atk1 == 1)[0], np.where(atk2 == 1)[0]))
            return np.random.choice(pool)
        else:
            empty = np.zeros(Board.BOARD_SIZE_SQ)
            x_min, x_max, y_min, y_max = oppo_board.move_area()
            for i in range(x_min, x_max + 1):
                for j in range(y_min, y_max + 1):
                    if oppo_board.stones[i*Board.BOARD_SIZE+j] == 0:
                        empty[i*Board.BOARD_SIZE+j] = 1
            pool = np.where(empty == 1)[0]
            return np.random.choice(pool)

    def oppo_selfplay(self):
        observation = self.make_observation_9_by_v(Board.STONE_WHITE)
        return dqn_selfplay.select_action(observation)

    def oppo_move(self):
        # if random.random() < 0.7:
        #     self.opponent.board = self.board.stones.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        #     _, row, col = self.opponent.search(self.board.STONE_WHITE, depth=1)
        #     self.board.move(Board.BOARD_SIZE * row + col, self.board.STONE_WHITE)
        # else:
        #     self.board.move(self.oppo_quick_play(), Board.STONE_WHITE)
        self.board.move(self.oppo_selfplay(), Board.STONE_WHITE)


