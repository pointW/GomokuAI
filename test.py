from env import Env



env = Env()
# observation, reward, done, _ = env.step(40)
# env.render()
# a = 1

env.board.move(84, 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
env.board.move(env.random_move(), 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
env.board.move(env.random_move(), 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
env.board.move(env.random_move(), 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
env.board.move(env.random_move(), 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
env.board.move(env.random_move(), 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
env.board.move(env.random_move(), 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
env.board.move(env.random_move(), 1)
env.board.move(env.oppo_2_move(), 2)
env.render()
a = 0

