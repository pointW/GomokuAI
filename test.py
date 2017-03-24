from env import Env
from board1 import Board

# env = Env()
# observation, reward, done, _ = env.step(40)
# env.render()
# a = 1

board = Board()
board.move1(4, 4, 1)
board.move1(6, 4, 2)
board.move1(5, 2, 1)
board.move1(5, 5, 2)
board.move1(3, 7, 1)
board.move1(4, 6, 2)
board.move1(3, 7, 6)
board.move1(4, 5, 2)
board.move1(3, 6, 1)
board.move1(3, 5, 2)
board.move1(6, 5, 1)
v1 = board.find_pattern()
board.move1(2, 5, 2)
v2 = board.find_pattern()
a = 1
