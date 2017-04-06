# -*- coding: utf-8 -*-

import numpy as np


class Board(object):
    STONE_EMPTY = 0
    STONE_BLACK = 1
    STONE_WHITE = 2
    WIN_STONE_NUM = 5
    WIN_PATTERN = {STONE_BLACK: np.ones(WIN_STONE_NUM, dtype=int) * STONE_BLACK,
                   STONE_WHITE: np.ones(WIN_STONE_NUM, dtype=int) * STONE_WHITE}
    BOARD_SIZE = 13
    BOARD_SIZE_SQ = BOARD_SIZE ** 2

    def __init__(self):
        self.stones = np.zeros(Board.BOARD_SIZE_SQ, np.int)
        self.over = False
        self.winner = Board.STONE_EMPTY
        self.last_move = -1

    def move(self, mv, v):
        self.stones[mv] = v
        self.last_move = mv

    def move1(self, x, y, v):
        self.move(x*Board.BOARD_SIZE+y, v)

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
        x_max = Board.BOARD_SIZE-1
        y_max = Board.BOARD_SIZE-1
        board = self.stones.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        for i in range(Board.BOARD_SIZE):
            if board[i, :].any():
                x_min = max(i-1, 0)
                break
        for i in range(Board.BOARD_SIZE):
            if board[:, i].any():
                y_min = max(i-1, 0)
                break
        for i in range(Board.BOARD_SIZE-1, -1, -1):
            if board[i, :].any():
                x_max = min(i+1, Board.BOARD_SIZE-1)
                break
        for i in range(Board.BOARD_SIZE-1, -1, -1):
            if board[:, i].any():
                y_max = min(i+1, Board.BOARD_SIZE-1)
                break
        return x_min, x_max, y_min, y_max

    # def find_pattern(self):
    #     board = self.stones.reshape(9, 9)
    #     mv = self.last_move
    #     c = self.stones[mv]
    #     x = mv / 9
    #     y = mv % 9
    #     b1 = False
    #     b2 = False
    #     empty = False
    #     count = 1
    #     # 横向右
    #     for i in range(x+1, 9):
    #         if i > 8:
    #             b1 = True
    #             break
    #         if board[i][y] == c:
    #             count += 1
    #             # 触边
    #             if i == 8:
    #                 b1 = True
    #         elif board[i][y] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b1 = True
    #             break
    #     # 横向左
    #     for i in range(x-1, -1, -1):
    #         if i < 0:
    #             b2 = True
    #             break
    #         if board[i][y] == c:
    #             count += 1
    #             if i == 0:
    #                 b2 = True
    #         elif board[i][y] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b2 = True
    #             break
    #     if count == 3 and b1 is False and b2 is False:
    #         return c, 3
    #     elif count == 4 and (b1 is False or b2 is False):
    #         return c, 4
    #
    #     count = 1
    #     b1 = False
    #     b2 = False
    #     empty = False
    #     # 纵向下
    #     for j in range(y+1, 9):
    #         if j > 8:
    #             b1 = True
    #             break
    #         if board[x][j] == c:
    #             count += 1
    #             if j == 8:
    #                 b1 = True
    #         elif board[x][j] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b1 = True
    #             break
    #     for j in range(y-1, -1, -1):
    #         if j < 0:
    #             b2 = True
    #             break
    #         if board[x][j] == c:
    #             count += 1
    #             if j == 0:
    #                 b2 = True
    #         elif board[x][j] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b2 = True
    #             break
    #     if count == 3 and b1 is False and b2 is False:
    #         return c, 3
    #     elif count == 4 and (b1 is False or b2 is False):
    #         return c, 4
    #
    #     count = 1
    #     b1 = False
    #     b2 = False
    #     empty = False
    #     for i in range(1, 9):
    #         if x+i > 8 or y+i > 8:
    #             b1 = True
    #             break
    #         if board[x+i][y+i] == c:
    #             count += 1
    #             if x+i == 8 or y+i == 8:
    #                 b1 = True
    #                 break
    #         elif board[x+i][y+i] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b1 = True
    #             break
    #     for i in range(1, 9):
    #         if x-i < 0 or y-i < 0:
    #             b2 = True
    #             break
    #         if board[x-i][y-i] == c:
    #             count += 1
    #             if x-i == 0 or y-i == 0:
    #                 b2 = True
    #                 break
    #         elif board[x-i][y-i] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b2 = True
    #             break
    #     if count == 3 and b1 is False and b2 is False:
    #         return c, 3
    #     elif count == 4 and (b1 is False or b2 is False):
    #         return c, 4
    #
    #     count = 1
    #     b1 = False
    #     b2 = False
    #     empty = True
    #     for i in range(1, 9):
    #         if x+i > 8 or y-i < 0:
    #             b1 = True
    #             break
    #         if board[x+i][y-i] == c:
    #             count += 1
    #             if x+i == 8 or y-i == 0:
    #                 b1 = True
    #                 break
    #         elif board[x+i][y-i] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b1 = True
    #             break
    #     for i in range(1, 9):
    #         if x-i <0 or y+i > 8:
    #             b2 = True
    #             break
    #         if board[x-i][y+i] == c:
    #             count += 1
    #             if x-i == 0 or y+i == 8:
    #                 b2 = True
    #                 break
    #         elif board[x-i][y+i] == self.STONE_EMPTY:
    #             if empty:
    #                 break
    #             empty = True
    #         else:
    #             b2 = True
    #             break
    #     if count == 3 and b1 is False and b2 is False:
    #         return c, 3
    #     elif count == 4 and (b1 is False or b2 is False):
    #         return c, 4
    #     return c, 0

    def find_pattern(self):
        board = self.stones.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE).copy()
        t_board = np.zeros((Board.BOARD_SIZE+2)*(Board.BOARD_SIZE+2), np.int).reshape(Board.BOARD_SIZE+2, Board.BOARD_SIZE+2)
        # add an edge of 3 around the board
        # for example:
        # 0 0 0 0 0
        # 0 0 1 2 0
        # 0 0 1 0 0
        # 0 0 1 0 0
        # 0 0 2 2 0
        # -->
        # 3 3 3 3 3 3 3
        # 3 0 0 0 0 0 3
        # 3 0 0 1 2 0 3
        # 3 0 0 1 0 0 3
        # 3 0 0 1 0 0 3
        # 3 0 0 2 2 0 3
        # 3 3 3 3 3 3 3
        for i in range(Board.BOARD_SIZE+2):
            for j in range(Board.BOARD_SIZE+2):
                if i in (0, Board.BOARD_SIZE+1) or j in (0, Board.BOARD_SIZE+1):
                    t_board[i][j] = 3
                else:
                    t_board[i][j] = board[i-1][j-1]
        board = t_board
        # for white, swap (1, 2)
        if self.stones[self.last_move] == self.STONE_WHITE:
            m_board = board.copy()
            for i in range(Board.BOARD_SIZE+2):
                for j in range(Board.BOARD_SIZE+2):
                    if m_board[i][j] == 1:
                        m_board[i][j] = 2
                    elif m_board[i][j] == 2:
                        m_board[i][j] = 1
            board = m_board

        mv = self.last_move
        # because of the edge, x and y have to += 1
        x = mv / Board.BOARD_SIZE + 1
        y = mv % Board.BOARD_SIZE + 1
        value = 0
        # horizonal 5s pattern
        for i in range(self.BOARD_SIZE-5+1+2):
            p = board[i:i+5, y]
            v, _ = self.get_dfs_value_offset(p)
            if v:
                value = max(v, value)
                break
        # horizonal 6s pattern
        for i in range(self.BOARD_SIZE-6+1+2):
            p = board[i:i+6, y]
            v, _ = self.get_dfs_value_offset(p)
            if v:
                value = max(v, value)
                break
        # vertical 5s pattern
        for j in range(self.BOARD_SIZE-5+1+2):
            p = board[x, j:j+5]
            v, _ = self.get_dfs_value_offset(p)
            if v:
                value = max(v, value)
                break
        # vertical 6s pattern
        for j in range(self.BOARD_SIZE-6+1+2):
            p = board[x, j:j+6]
            v, _ = self.get_dfs_value_offset(p)
            if v:
                value = max(v, value)
                break
        # compute the length of the diagonal
        d = y-x
        if d >= 0:
            l = Board.BOARD_SIZE+2-d
        else:
            l = Board.BOARD_SIZE+2+d
        # diagonal 5s pattern
        if l >= 5:
            p = board.diagonal(d)
            for i in range(l-5+1):
                v, _ = self.get_dfs_value_offset(p[i:i+5])
                if v:
                    value = max(v, value)
                    break
        # diagonal 6s pattern
        if l >= 6:
            p = board.diagonal(d)
            for i in range(l-6+1):
                v, _ = self.get_dfs_value_offset(p[i:i+6])
                if v:
                    value = max(v, value)
                    break
        # swap rows to compute another diagonal
        m_board = board.copy()
        for i in range(Board.BOARD_SIZE+2):
            m_board[i] = board[Board.BOARD_SIZE+1-i]
        # compute the length of the diagonal
        x = Board.BOARD_SIZE+1-x
        d = y-x
        if d >= 0:
            l = Board.BOARD_SIZE+2-d
        else:
            l = Board.BOARD_SIZE+2+d
        # patterns of another diagonal
        if l >= 5:
            p = m_board.diagonal(d)
            for i in range(l-5+1):
                v, _ = self.get_dfs_value_offset(p[i:i+5])
                if v:
                    value = max(v, value)
                    break
        if l >= 6:
            p = m_board.diagonal(d)
            for i in range(l-6+1):
                v, _ = self.get_dfs_value_offset(p[i:i+6])
                if v:
                    value = max(v, value)
                    break
        return value

    # @staticmethod
    # def cal_value(p):
    #     if len(p) == 5:
    #         if p.sum() == 3 and p.max() == 1:
    #             # x_x_x, xx__x
    #             if p[0] == 1 and p[4] == 1:
    #                 return 0, None
    #             # __xxx
    #             elif p[0] == 0 and p[1] == 0:
    #                 return 0, None
    #             # xxx__
    #             elif p[3] == 0 and p[4] == 0:
    #                 return 0, None
    #             # _x_xx
    #             elif p[2] == 0 and (p[0] == 0 or p[4] == 0):
    #                 return 0, None
    #             # _xxx_
    #             elif p[1:4].sum() == 3:
    #                 return 1, [0, 4]
    #         elif p.sum() == 4 and p.max() == 1:
    #             # xx_xx, x_xxx
    #             if p[0] and p[4]:
    #                 return 2, [p.argmin()]
    #         else:
    #             return 0, None
    #
    #     elif len(p) == 6:
    #         # _xxxx_
    #         if p.sum() == 4 and p.max() == 1 and p[0] == 0 and p[5] == 0:
    #             return 5, None
    #         # oxxxx_, |xxxx_
    #         elif p[1:5].sum() == 4 and p[1:5].max() == 1 and (p[0] == 0 or p[5] == 0):
    #             return 2, [p.argmin()]
    #         # _x_xx_
    #         elif p[1] == 1 and p[4] == 1 and p[0] == 0 and p[5] == 0 and p[1:5].sum() == 3:
    #             return 1, [i for i, v in enumerate(p.tolist()) if v == 0]
    #         else:
    #             return 0, None
    #
    #     return 0, None

    # @staticmethod
    # def get_value_and_offset(p):
    #     if len(p) == 5:
    #         if p.sum() == 4 and p.max() == 1:
    #             # xx_xx, x_xxx
    #             if p[0] and p[4]:
    #                 return 4, [p.argmin()]
    #     elif len(p) == 6:
    #         if p.sum() == 2 and p.max() == 1:
    #             # __xx__
    #             if p[2] and p[3]:
    #                 return 2, [1, 4]
    #             # _x_x__
    #             elif p[1] and p[3]:
    #                 return 2, [2, 4]
    #             # __x_x_
    #             elif p[2] and p[4]:
    #                 return 2, [1, 3]
    #         elif p.sum() == 3 and p.max() == 1:
    #             # _xxx__
    #             if p[1] and p[2] and p[3]:
    #                 return 3, [4]
    #             # __xxx_
    #             elif p[2] and p[3] and p[4]:
    #                 return 3, [1]
    #             # _x_xx_
    #             elif p[1] and p[3] and p[4]:
    #                 return 3, [0, 2, 5]
    #             # _xx_x_
    #             elif p[1] and p[2] and p[4]:
    #                 return 3, [0, 3, 5]
    #         # _xxxx_
    #         elif p.sum() == 4 and p.max() == 1 and not p[0] and not p[5]:
    #             return 5, [0, 5]
    #         # oxxxx_, |xxxx_, r
    #         elif p[1:5].sum() == 4 and p[1:5].max() == 1 and (p[0] == 0 or p[5] == 0):
    #             return 4, [p.argmin()]
    #     return 0, []

    @staticmethod
    def get_atk_value_offset(p):
        if len(p) == 5:
            # xx_xx, x_xxx, _xxxx
            if p.sum() == 4 and p.max() == 1:
                return 5, [p.argmin()]
        elif len(p) == 6:
            if p.sum() == 3 and p.max() == 1:
                # _xxx__
                if p[1] and p[2] and p[3]:
                    return 4, [4]
                # __xxx_
                if p[2] and p[3] and p[4]:
                    return 4, [1]
                # _x_xx_
                if p[1] and p[3] and p[4]:
                    return 4, [2]
                # _xx_x_
                if p[1] and p[2] and p[4]:
                    return 4, [3]
            # o_xxx_, oxxx__, o__xxx, oxx_x_, ox_xx_, o_x_xx, o_xx_x, r
            elif (p[1:6].sum() == 3 and p[1:6].max() == 1)\
                    or (p[0:5].sum() == 3 and p[0:5].max() == 1):
                return 3, np.where(p == 0)[0].tolist()
            elif p.sum() == 2 and p.max() == 1:
                # _xx___
                if p[1] and p[2]:
                    return 2, [3, 4]
                # __xx__
                elif p[2] and p[3]:
                    return 2, [1, 4]
                # ___xx_
                elif p[3] and p[4]:
                    return 2, [1, 2]
                # _x_x__
                elif p[1] and p[3]:
                    return 2, [2, 4]
                # __x_x_
                elif p[2] and p[4]:
                    return 2, [1, 3]
        return 0, []

    @staticmethod
    def get_dfs_value_offset(p):
        if len(p) == 5:
            if p.sum() == 4 and p.max() == 1:
                # xx_xx, x_xxx
                if p[0] and p[4]:
                    return 4, [p.argmin()]
        elif len(p) == 6:
            if p.sum() == 3 and p.max() == 1:
                # _xxx__
                if p[1] and p[2] and p[3]:
                    return 3, [0, 4]
                # __xxx_
                elif p[2] and p[3] and p[4]:
                    return 3, [1, 5]
                # _x_xx_
                elif p[1] and p[3] and p[4]:
                    return 3, [0, 2, 5]
                # _xx_x_
                elif p[1] and p[2] and p[4]:
                    return 3, [0, 3, 5]
            # _xxxx_
            elif p.sum() == 4 and p.max() == 1 and not p[0] and not p[5]:
                return 5, []
            # oxxxx_, |xxxx_, r
            elif p[1:5].sum() == 4 and p[1:5].max() == 1 and (p[0] == 0 or p[5] == 0):
                return 4, [p.argmin()]
        return 0, []

    def gen_dfs_atk_moves(self, atk):
        board = self.stones.reshape(Board.BOARD_SIZE, Board.BOARD_SIZE).copy()
        t_board = np.zeros((Board.BOARD_SIZE+2) * (Board.BOARD_SIZE+2), np.int).reshape(Board.BOARD_SIZE+2, Board.BOARD_SIZE+2)
        # add an edge of 3 around the board
        # for example:
        # 0 0 0 0 0
        # 0 0 1 2 0
        # 0 0 1 0 0
        # 0 0 1 0 0
        # 0 0 2 2 0
        # -->
        # 3 3 3 3 3 3 3
        # 3 0 0 0 0 0 3
        # 3 0 0 1 2 0 3
        # 3 0 0 1 0 0 3
        # 3 0 0 1 0 0 3
        # 3 0 0 2 2 0 3
        # 3 3 3 3 3 3 3
        for i in range(Board.BOARD_SIZE+2):
            for j in range(Board.BOARD_SIZE+2):
                if i in (0, Board.BOARD_SIZE+1) or j in (0, Board.BOARD_SIZE+1):
                    t_board[i][j] = 3
                else:
                    t_board[i][j] = board[i - 1][j - 1]
        board = t_board
        # when dfs, swap (1, 2)
        if not atk:
            m_board = board.copy()
            for i in range(Board.BOARD_SIZE+2):
                for j in range(Board.BOARD_SIZE+2):
                    if m_board[i][j] == 1:
                        m_board[i][j] = 2
                    elif m_board[i][j] == 2:
                        m_board[i][j] = 1
            board = m_board

        level2 = np.zeros(Board.BOARD_SIZE*Board.BOARD_SIZE, np.int).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        level3 = np.zeros(Board.BOARD_SIZE*Board.BOARD_SIZE, np.int).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        level4 = np.zeros(Board.BOARD_SIZE*Board.BOARD_SIZE, np.int).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)
        level5 = np.zeros(Board.BOARD_SIZE*Board.BOARD_SIZE, np.int).reshape(Board.BOARD_SIZE, Board.BOARD_SIZE)

        # horizonal and vertical
        for i in range(self.BOARD_SIZE+2):
            for j in range(self.BOARD_SIZE+2):
                p = board[i, j:j+5]
                if atk:
                    v, positions = self.get_atk_value_offset(p)
                else:
                    v, positions = self.get_dfs_value_offset(p)
                if v == 2:
                    for position in positions:
                        level2[i-1, j+position-1] = 1
                if v == 3:
                    for position in positions:
                        level3[i-1, j+position-1] = 1
                if v == 4:
                    for position in positions:
                        level4[i-1, j+position-1] = 1
                if v == 5:
                    for position in positions:
                        level5[i-1, j+position-1] = 1

                p = board[i:i+5, j]
                if atk:
                    v, positions = self.get_atk_value_offset(p)
                else:
                    v, positions = self.get_dfs_value_offset(p)
                if v == 2:
                    for position in positions:
                        level2[i+position-1, j-1] = 1
                if v == 3:
                    for position in positions:
                        level3[i+position-1, j-1] = 1
                if v == 4:
                    for position in positions:
                        level4[i+position-1, j-1] = 1
                if v == 5:
                    for position in positions:
                        level5[i+position-1, j-1] = 1

                p = board[i, j:j+6]
                if atk:
                    v, positions = self.get_atk_value_offset(p)
                else:
                    v, positions = self.get_dfs_value_offset(p)
                if v == 2:
                    for position in positions:
                        level2[i-1, j+position-1] = 1
                if v == 3:
                    for position in positions:
                        level3[i-1, j+position-1] = 1
                if v == 4:
                    for position in positions:
                        level4[i-1, j+position-1] = 1
                if v == 5:
                    for position in positions:
                        level5[i-1, j+position-1] = 1

                p = board[i:i+6, j]
                if atk:
                    v, positions = self.get_atk_value_offset(p)
                else:
                    v, positions = self.get_dfs_value_offset(p)
                if v == 2:
                    for position in positions:
                        level2[i+position-1, j-1] = 1
                if v == 3:
                    for position in positions:
                        level3[i+position-1, j-1] = 1
                if v == 4:
                    for position in positions:
                        level4[i+position-1, j-1] = 1
                if v == 5:
                    for position in positions:
                        level5[i+position-1, j-1] = 1

        # diagonal
        for offset in range(-(Board.BOARD_SIZE-5), (Board.BOARD_SIZE-5)+1):
            p = board.diagonal(offset)
            for i in range(Board.BOARD_SIZE+2-offset-5+1):
                if atk:
                    v, positions = self.get_atk_value_offset(p[i:i+5])
                else:
                    v, positions = self.get_dfs_value_offset(p[i:i+5])
                # v, positions = self.get_value_and_offset(p[i:i+5])
                if v == 2:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level2[x, y] = 1
                if v == 3:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level3[x, y] = 1
                if v == 4:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level4[x, y] = 1
                if v == 5:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level5[x, y] = 1
                if atk:
                    v, positions = self.get_atk_value_offset(p[i:i+6])
                else:
                    v, positions = self.get_dfs_value_offset(p[i:i+6])
                # v, positions = self.get_value_and_offset(p[i:i+6])
                if v == 2:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level2[x, y] = 1
                if v == 3:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level3[x, y] = 1
                if v == 4:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level4[x, y] = 1
                if v == 5:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level5[x, y] = 1

        # swap rows to compute another diagonal
        m_board = board.copy()
        for i in range(Board.BOARD_SIZE+2):
            m_board[i] = board[Board.BOARD_SIZE+1 - i]
        # another diagonal
        for offset in range(-(Board.BOARD_SIZE-5), (Board.BOARD_SIZE-5)+1):
            p = m_board.diagonal(offset)
            for i in range(Board.BOARD_SIZE+2-offset-5+1):
                if atk:
                    v, positions = self.get_atk_value_offset(p[i:i+5])
                else:
                    v, positions = self.get_dfs_value_offset(p[i:i+5])
                # v, positions = self.get_value_and_offset(p[i:i+5])
                if v == 2:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level2[Board.BOARD_SIZE-1-x, y] = 1
                if v == 3:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level3[Board.BOARD_SIZE-1-x, y] = 1
                if v == 4:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level4[Board.BOARD_SIZE-1-x, y] = 1
                if v == 5:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level5[Board.BOARD_SIZE-1-x, y] = 1
                if atk:
                    v, positions = self.get_atk_value_offset(p[i:i+6])
                else:
                    v, positions = self.get_dfs_value_offset(p[i:i+6])
                # v, positions = self.get_value_and_offset(p[i:i+6])
                if v == 2:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level2[Board.BOARD_SIZE-1-x, y] = 1
                if v == 3:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level3[Board.BOARD_SIZE-1-x, y] = 1
                if v == 4:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level4[Board.BOARD_SIZE-1-x, y] = 1
                if v == 5:
                    for position in positions:
                        x, y = self.index_by_diagonal(offset, i, position)
                        level5[Board.BOARD_SIZE-1-x, y] = 1
        if atk:
            return level2, level3, level4, level5
        else:
            return level3, level4

    @staticmethod
    def index_by_diagonal(offset, i, position):
        if offset >= 0:
            return i+position-1, offset+i+position-1
        if offset < 0:
            return 0-offset+i+position-1, i+position-1

    def reverse(self):
        for i in range(self.stones.size):
            if self.stones[i] == 1:
                self.stones[i] = 2
            elif self.stones[i] == 2:
                self.stones[i] = 1
