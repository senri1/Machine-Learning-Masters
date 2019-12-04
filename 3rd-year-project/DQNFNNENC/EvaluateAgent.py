#from LinearAgent import Linearagent
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
import numpy as np
import os
import torch
from LinearAgent import Linearagent


def collectRandomData(replayMemory,steps,env_name):
    env = make_atari(env_name)
    env = wrap_deepmind(env)
    i=0
    while i in range(steps):
        
        done = False
        initial_state = env.reset()
        action = np.random.randint(0,high=4) 
        state, reward, done, _ = env.step(action)
        replayMemory.add(initial_state,action,reward,state,done )
        i += 1
    
        while (not done) and (i < steps) :
        
            action = np.random.randint(0,high=4)
            next_state,reward,done,_ = env.step(action)
            replayMemory.add(state,action,reward,next_state,done)
            state = next_state
            i += 1
        print(i)
        
    env.close()

def collectMeanScore(agent,steps,epsilon,env_name):
    env = make_atari(env_name)
    env = wrap_deepmind(env)
    evalAgent = agent
    evalAgent.epsilon = epsilon
    rewards_sum = 0.0
    episodes = 0

    state = env.reset()
    while episodes in range(steps):
        
        action = evalAgent.getAction(LazyFrame2Torch(state))
        state, reward, done, _ = env.step(action)
        rewards_sum += reward
        if done:
            env.reset()
            print(episodes)
            episodes += 1

    average_score = rewards_sum/episodes
    env.close()
    return float(average_score)

def evaluateStateQvalues(agent):
    
    s = np.load('log/stateEval.npy')
    with torch.no_grad():
        s = torch.from_numpy(s).float().to('cuda')
        s = agent.encoder(s)
        s = s.view(-1,400)
        q = agent.fnn(s).detach().cpu().numpy()
    q = np.mean(q,axis=1)
    q = np.mean(q,axis=0)
    return float(q)



def evaluate():

    TestAgent = Linearagent()
    evaluation_data=[]
    idx = int(np.load('log/idx.npy'))
    env_name = 'Breakout-v0'

    for i in range(240):

        print(i)
        TestAgent.load_agent(str(i))            
        evaluation_data.append([ TestAgent.training_steps, collectMeanScore(TestAgent,25,0.01,env_name), evaluateStateQvalues(TestAgent) ])

    np.savetxt("log/eval_data.csv", evaluation_data, delimiter=",")

def LazyFrame2Torch(x):
        y = x.__array__()[np.newaxis,:,:,:]
        y = np.moveaxis(y,3,1)
        y = torch.from_numpy(y).float().to('cuda')
        return y






