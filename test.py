from env import Env
from board1 import Board
from pg import PolicyNetwork
import torch
from torch.autograd import Variable


env = Env()
# observation, reward, done, _ = env.step(40)
# env.render()
# a = 1

model = PolicyNetwork()
model.type(torch.FloatTensor)
state = torch.from_numpy(env.make_observation_7())
state = state.type(torch.FloatTensor)
state = state.unsqueeze(0)
states = []
states.append(state)
states.append(state)
states = torch.cat(states)
a = model(Variable(states)).data
print a

