import gym
import gym_gomoku

env = gym.make('Gomoku9x9-v0') # default 'beginner' level opponent policy
env.reset()
env.render()

# play a game
env.reset()
for _ in range(20):
    action = env.action_space.sample() # sample without replacement
    observation, reward, done, info = env.step(action)
    print reward
    env.render()
    if done:
        break
        print ("Game is Over")
