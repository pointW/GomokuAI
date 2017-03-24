from board1 import Board
from opponent import Searcher
import numpy as np


class Env(object):
    board = Board()
    opponent = Searcher()
    reward = 0
    BLACK = Board.STONE_BLACK
    WHITE = Board.STONE_WHITE
    EMPTY = Board.STONE_EMPTY

    def reset(self):
        self.board = Board()
        self.opponent = Searcher()

    def step(self, mv):
        self.board.move(mv, self.board.STONE_BLACK)
        done, _ = self.board.is_over()
        if done:
            observation = self.board.stones.reshape(9, 9)
            self.reward = 10
            _ = None
            return observation, self.reward, done, _
        else:
            self.reward = self.get_reward()
            self.opponent.board = self.board.stones.reshape(9, 9)
            _, row, col = self.opponent.search(self.board.STONE_WHITE, depth=2)
            self.board.move(9*row+col, self.board.STONE_WHITE)
            done, _ = self.board.is_over()
            observation = self.board.stones.reshape(9, 9)
            _ = None
            if done:
                self.reward = -10
            else:
                new_reward = self.get_reward()
                if new_reward != 0:
                    self.reward = new_reward
            return observation, self.reward, done, _

    def render(self):
        b = self.board.stones.reshape(9, 9)
        p = ''
        for i in range(9):
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
        print 'reward = %d' % self.reward

    def get_reward(self):
        c, p = self.board.find_pattern()
        if c == self.board.STONE_BLACK and (p == 3 or p == 4):
            return 0.5
        if c == self.board.STONE_WHITE and (p == 3 or p == 4):
            return -5
        return 0

