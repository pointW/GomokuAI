# -*- coding: utf-8 -*-

import numpy as np


class Board(object):
    STONE_EMPTY = 0
    STONE_BLACK = 1
    STONE_WHITE = 2
    WIN_STONE_NUM = 5
    WIN_PATTERN = {STONE_BLACK: np.ones(WIN_STONE_NUM, dtype=int) * STONE_BLACK,
                   STONE_WHITE: np.ones(WIN_STONE_NUM, dtype=int) * STONE_WHITE}
    BOARD_SIZE = 9
    BOARD_SIZE_SQ = BOARD_SIZE ** 2

    def __init__(self):
        self.stones = np.zeros(Board.BOARD_SIZE_SQ, np.int)
        self.over = False
        self.winner = Board.STONE_EMPTY
        self.last_move = -1

    def move(self, mv, v):
        self.stones[mv] = v
        self.last_move = mv

    @staticmethod
    def _row(arr2d, row, col):
        return arr2d[row, :]

    @staticmethod
    def _col(arr2d, row, col):
        return arr2d[:, col]

    @staticmethod
    def _diag(arr2d, row, col):
        return np.diag(arr2d, col - row)

    @staticmethod
    def _diag_counter(arr2d, row, col):
        return Board._diag(np.rot90(arr2d), arr2d.shape[1] - 1 - col, row)

    @staticmethod
    def _find_subseq(seq, sub):
        '''
        Returns:
        ---------------
        indexes: array
            all occurs of sub in seq
        '''
        #         print('sub seq find:')
        #         print(seq)
        #         print(sub)

        assert seq.size >= sub.size

        target = np.dot(sub, sub)
        candidates = np.where(np.correlate(seq, sub) == target)[0]
        # some of the candidates entries may be false positives, double check
        check = candidates[:, np.newaxis] + np.arange(len(sub))
        mask = np.all((np.take(seq, check) == sub), axis=-1)
        return candidates[mask]

    def find_conn_5(self, board, center_row, center_col, who):
        lines = []
        lines.append(Board._row(board, center_row, center_col))
        lines.append(Board._col(board, center_row, center_col))
        lines.append(Board._diag(board, center_row, center_col))
        lines.append(Board._diag_counter(board, center_row, center_col))
        for v in lines:
            if v.size < Board.WIN_STONE_NUM:
                continue
            occur = Board._find_subseq(v, Board.WIN_PATTERN[who])
            if occur.size != 0:
                return True
        return False

    @staticmethod
    def find_pattern_will_win(board, who):
        pats = np.identity(Board.WIN_STONE_NUM, int)
        pats = 1 - pats
        pats[pats == 1] = who

        board = board.stones.reshape(-1, Board.BOARD_SIZE)

        lines = []
        for i in range(Board.BOARD_SIZE):
            lines.append(Board._row(board, i, 0))
            lines.append(Board._col(board, 0, i))
            lines.append(Board._diag(board, i, 0))
            lines.append(Board._diag(board, 0, i))
            lines.append(Board._diag_counter(board, i, Board.BOARD_SIZE - 1))
            lines.append(Board._diag_counter(board, 0, i))

        for v in lines:
            if v.size < Board.WIN_STONE_NUM:
                continue
            for p in pats:
                occur = Board._find_subseq(v, p)
                if occur.size != 0:
                    return True

        return False

    @staticmethod
    def find_conn_5_all(board):
        lines = []
        for i in range(Board.BOARD_SIZE):
            lines.append(Board._row(board, i, 0))
            lines.append(Board._col(board, 0, i))
            lines.append(Board._diag(board, i, 0))
            lines.append(Board._diag(board, 0, i))
            lines.append(Board._diag_counter(board, i, Board.BOARD_SIZE - 1))
            lines.append(Board._diag_counter(board, 0, i))
        for v in lines:
            if v.size < Board.WIN_STONE_NUM:
                continue
            occur = Board._find_subseq(v, Board.WIN_PATTERN[Board.STONE_BLACK])
            if occur.size != 0:
                return True
            occur = Board._find_subseq(v, Board.WIN_PATTERN[Board.STONE_WHITE])
            if occur.size != 0:
                return True

        return False

    def is_over(self):
        loc = self.last_move
        who = self.stones[loc]
        grid = self.stones.reshape(-1, Board.BOARD_SIZE)
        row, col = divmod(loc, Board.BOARD_SIZE)

        #         print('who[%d] at [%d, %d]' % (who, row, col))
        #         print(grid)

        win = self.find_conn_5(grid, row, col, who)
        if win:
            self.over = True
            self.winner = who
            return True, who

        if np.where(self.stones == 0)[0].size == 0:  # the last step
            self.over = True
            return True, Board.STONE_EMPTY

        return False, None

    def move_area(self):
        x_min = 0
        y_min = 0
        x_max = 8
        y_max = 8
        board = self.stones.reshape(9, 9)
        for i in range(8):
            if board[i, :].any():
                x_min = max(i-2, 0)
                break
        for i in range(8):
            if board[:, i].any():
                y_min = max(i-2, 0)
                break
        for i in range(8, -1, -1):
            if board[i, :].any():
                x_max = min(i+2, 8)
                break
        for i in range(8, -1, -1):
            if board[:, i].any():
                y_max = min(i+2, 8)
                break
        return x_min, x_max, y_min, y_max

    def find_pattern(self):
        board = self.stones.reshape(9, 9)
        mv = self.last_move
        c = self.stones[mv]
        x = mv / 9
        y = mv % 9
        b1 = False
        b2 = False
        empty = False
        count = 1
        # 横向右
        for i in range(x+1, 9):
            if i > 8:
                b1 = True
                break
            if board[i][y] == c:
                count += 1
                # 触边
                if i == 8:
                    b1 = True
            elif board[i][y] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b1 = True
                break
        # 横向左
        for i in range(x-1, -1, -1):
            if i < 0:
                b2 = True
                break
            if board[i][y] == c:
                count += 1
                if i == 0:
                    b2 = True
            elif board[i][y] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b2 = True
                break
        if count == 3 and b1 is False and b2 is False:
            return c, 3
        elif count == 4 and (b1 is False or b2 is False):
            return c, 4

        count = 1
        b1 = False
        b2 = False
        empty = False
        # 纵向下
        for j in range(y+1, 9):
            if j > 8:
                b1 = True
                break
            if board[x][j] == c:
                count += 1
                if j == 8:
                    b1 = True
            elif board[x][j] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b1 = True
                break
        for j in range(y-1, -1, -1):
            if j < 0:
                b2 = True
                break
            if board[x][j] == c:
                count += 1
                if j == 0:
                    b2 = True
            elif board[x][j] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b2 = True
                break
        if count == 3 and b1 is False and b2 is False:
            return c, 3
        elif count == 4 and (b1 is False or b2 is False):
            return c, 4

        count = 1
        b1 = False
        b2 = False
        empty = False
        for i in range(1, 9):
            if x+i > 8 or y+i > 8:
                b1 = True
                break
            if board[x+i][y+i] == c:
                count += 1
                if x+i == 8 or y+i == 8:
                    b1 = True
                    break
            elif board[x+i][y+i] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b1 = True
                break
        for i in range(1, 9):
            if x-i < 0 or y-i < 0:
                b2 = True
                break
            if board[x-i][y-i] == c:
                count += 1
                if x-i == 0 or y-i == 0:
                    b2 = True
                    break
            elif board[x-i][y-i] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b2 = True
                break
        if count == 3 and b1 is False and b2 is False:
            return c, 3
        elif count == 4 and (b1 is False or b2 is False):
            return c, 4

        count = 1
        b1 = False
        b2 = False
        empty = True
        for i in range(1, 9):
            if x+i > 8 or y-i < 0:
                b1 = True
                break
            if board[x+i][y-i] == c:
                count += 1
                if x+i == 8 or y-i == 0:
                    b1 = True
                    break
            elif board[x+i][y-i] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b1 = True
                break
        for i in range(1, 9):
            if x-i <0 or y+i > 8:
                b2 = True
                break
            if board[x-i][y+i] == c:
                count += 1
                if x-i == 0 or y+i == 8:
                    b2 = True
                    break
            elif board[x-i][y+i] == self.STONE_EMPTY:
                if empty:
                    break
                empty = True
            else:
                b2 = True
                break
        if count == 3 and b1 is False and b2 is False:
            return c, 3
        elif count == 4 and (b1 is False or b2 is False):
            return c, 4
        return c, 0