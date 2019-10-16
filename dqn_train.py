import math
import random
import os
import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn import TetrisDQN
from replay import ReplayMemory
from tetris.tetris import Tetris

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# parameters
batchSize = 128
gamma = 0.999
exploreStart = 0.9
exploreEnd = 0.05
exploreDecay = 20000
explore = exploreStart
numStepPerUpdate = 1000
memoryCapacity = 10000
numEpisodes = 10000
maxStepPerEpisode = 10000

# rendering
render = True
renderStepDuration = 50

# initialization
policyNet = TetrisDQN()
targetNet = TetrisDQN()
policyNet.to(device)
targetNet.to(device)
memory = ReplayMemory(memoryCapacity)
tetris = Tetris()
optimizer = optim.RMSprop(policyNet.parameters(), lr=1)

numSteps = 0
done = True

# save results
currentTime = datetime.datetime.now()
timeString = currentTime.strftime("%Y-%m-%d-%H-%M-%S")
if not os.path.exists("results"):
    os.mkdir("results")
directory = "results/" + timeString + "/"
os.mkdir(directory)
configFile = open(directory + "config.txt", "w")
configFile.writelines([
    "batchSize: %d\n" % batchSize,
    "gamma: %f\n" % gamma,
    "exploreStart: %f\n" % exploreStart,
    "exploreEnd: %f\n" % exploreEnd,
    "exploreDecay: %f\n" % exploreDecay,
    "numStepsPerUpdate: %d\n" % numStepPerUpdate,
    "memoryCapacity: %d\n" % memoryCapacity,
    "maxStepPerEpisode: %d\n" % maxStepPerEpisode
])
configFile.close()

def select_action(state):
    global explore
    r = random.random()
    explore = exploreEnd + (exploreStart - exploreEnd) * math.exp(-1 * numSteps / exploreDecay)
    if r > explore:
        with torch.no_grad():
            return policyNet(torch.tensor([state], device=device, dtype=torch.float)).max(1)[1].view(1, 1)
    else:
        return random.randint(0, tetris.num_actions - 1)

def train():
    if len(memory) < batchSize:
        return
    transitions = memory.sample(batchSize)
    states, actions, rewards, nextStates = zip(*transitions)
    states = torch.tensor(states, device=device, dtype=torch.float)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    nonFinalMask = torch.tensor([s is not None for s in nextStates], device=device, dtype=torch.bool)
    nonFinalNextStates = torch.tensor([s for s in nextStates if s is not None], device=device, dtype=torch.float)

    qAll = policyNet(states)
    q = qAll.gather(1, actions.unsqueeze(1)).squeeze()
    qNext = torch.zeros(batchSize, device=device, dtype=torch.float)
    qNext[nonFinalMask] = targetNet(nonFinalNextStates).max(1)[0].detach()
    qExpect = qNext * gamma + rewards

    loss = F.smooth_l1_loss(q, qExpect)

    optimizer.zero_grad()
    loss.backward()
    # for param in policyNet.parameters():
    #     param.grad.data.clamp(-1, 1)
    optimizer.step()


### main
for episode in range(numEpisodes):

    # episodeDirectory = directory + "ep%d/" % episode
    # os.mkdir(episodeDirectory)

    tetris.start()
    state = tetris.get_state()
    episodeReward = 0
    step = 0
    while step < maxStepPerEpisode:
        
        # if render and numSteps % renderStepDuration == 0:
        #     image = tetris.get_printed_state()
        #     plt.imshow(image)
        #     plt.savefig(episodeDirectory + "s%d.jpg" % step)

        action = select_action(state)
        step += 1
        next_state, reward, done = tetris.step(action)
        if done:
            next_state = None
        memory.add((state, action, reward, next_state))
        state = next_state
        episodeReward += reward

        train()

        numSteps += 1
        if numSteps % numStepPerUpdate == 0:
            targetNet.load_state_dict(policyNet.state_dict())

        if done:
            break

    # if render:
    #     image = tetris.get_printed_state()
    #     plt.imshow(image)
    #     plt.savefig(episodeDirectory + "s%d.jpg" % step)
    #     episodeFile = open(episodeDirectory + "result.txt", "w")
    #     episodeFile.write("Steps: %d, Rewards: %d, Total Steps: %d, Ending Explore: %f" % (step, episodeReward, numSteps, explore))
    #     episodeFile.close()
    
    image = tetris.get_printed_state()
    plt.imshow(image)
    plt.savefig(directory + "e%ds%dr%d.jpg" % (episode, step, episodeReward))

    print("Episode: %d, Steps: %d/%d, Explore: %.3f, Reward: %d" % (episode, step, numSteps, explore, episodeReward))

