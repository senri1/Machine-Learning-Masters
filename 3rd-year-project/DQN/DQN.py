import numpy as np 
import gym
import torch
import torch.nn.functional as F
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from DQNAgent import DQNagent
from ReplayMemory import ReplayMemory
from EvaluateAgent import LazyFrame2Torch, collectRandomData, collectMeanScore
import time
from datetime import timedelta
import os
from mem_top import mem_top
from collections import defaultdict
from gc import get_objects

env_name = 'Breakout-v0'
env = make_atari(env_name)
env = wrap_deepmind(env)


frames = 8000
episodes = 0
batch_size = 32
memory_size = 1000
memory_start_size = int(memory_size/20)
update_frequency = 10000
evaluation_frequency = frames/250

memory = ReplayMemory(memory_size, batch_size)
agent = DQNagent()
collectRandomData(memory,memory_start_size,env_name)

#agent.load_agent('FINAL')
#memory.load_replay('FINAL')
print('Training for ' + str(frames) + ' frames.')
print('Batch size = ' , batch_size)
print('Initial memory size = ' , memory.current_size)
print('Update Q target frequency = ', update_frequency)
print('Evaluation frequency = ' , evaluation_frequency)

n = 0
j = agent.training_steps
"""print(mem_top()) #5362
before = defaultdict(int)
after = defaultdict(int)
for i in get_objects():
    before[type(i)] += 1 """

try:
    while n in range(frames):

        done = False
        initial_state = env.reset()
        action = agent.getAction(LazyFrame2Torch(initial_state)) 
        state, reward, done, _ = env.step(action)
        memory.add(initial_state,action,reward,state,done )
        agent.decrease_epsilon()
        n += 1
            
        while (not done) and (n<frames):

            action = agent.getAction(LazyFrame2Torch(state)) 
            next_state,reward,done,_ = env.step(action)
            memory.add(state,action,reward,next_state,done)
            state = next_state
            agent.decrease_epsilon()
            n += 1

            if memory.current_size >= batch_size:

                state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch()
                qtargets = agent.getQtargets(next_state_batch, reward_batch, not_done_batch)
                agent.train(state_batch, action_batch, batch_size, qtargets)
                

            if n % update_frequency == 0:
                #start_time = time.monotonic()
                agent.update_target()
                #end_time = time.monotonic()
                #print('Block 4 time: ',timedelta(seconds=end_time - start_time))
            

            if n % evaluation_frequency == 0:
                agent.save_agent(j)
                j+=1
                print('Current epsilon: ',agent.epsilon)
                print('Frames = ', agent.training_steps)
                print('Number of episodes = ', episodes)
                print('Number of saved agents = ',j)
                

    
            
        episodes += 1
        

except:
    print('Welp')


agent.save_agent('FINAL')
memory.save_replay('FINAL')
np.save('log/idx',j)
print('Final avergae score: ', collectMeanScore(agent,5,0.005,env_name))
print("Total number of frames: ", frames)
print("Total number of episodes: ", episodes)
