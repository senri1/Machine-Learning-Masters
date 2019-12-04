import numpy as np 
import gym
import torch
import torch.nn.functional as F
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from LinearAgent import Linearagent
from ReplayMemory import ReplayMemory
from EvaluateAgent import collectRandomData, LazyFrame2Torch, collectMeanScore
import time
from datetime import timedelta
import os
from mem_top import mem_top
from collections import defaultdict
from gc import get_objects


env_name = 'Breakout-v0'
env = make_atari(env_name)
env = wrap_deepmind(env)

frames = 2000000
episodes = 0
batch_size = 32
memory_size = 250000
memory_start_size = int(memory_size/20)
update_frequency = 10000
evaluation_frequency = frames/250

memory = ReplayMemory(memory_size, batch_size)
agent = Linearagent()
collectRandomData(memory,memory_start_size,env_name)
agent.load_agent('one')

print('Training for number of frames = ' ,str(frames))
print('Batch size = ' , batch_size)
print('Initial memory size = ' , len(memory.data))
print('Update Q target frequency = ', update_frequency)
print('Evaluation frequency = ' , evaluation_frequency)

n = 0
j = 0
print(mem_top()) #5362

try:
    while n in range(frames):

        done = False
        initial_state = env.reset()
        action = agent.getAction(LazyFrame2Torch(initial_state)) 
        state, reward, done, _ = env.step(action)
        memory.add(initial_state,action,reward,state,done )
        agent.decrease_epsilon()
        n += 1  #5367
            
        while (not done):

            action = agent.getAction(LazyFrame2Torch(state)) 
            next_state,reward,done,_ = env.step(action)
            memory.add(state,action,reward,next_state,done)
            state = next_state
            agent.decrease_epsilon()
            n += 1  #5368
           
            if memory.current_size >= batch_size:

                # get batch of size 32 from replay memory
                state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch()
                # get qtargets
                qtargets = agent.getQtargets(next_state_batch, reward_batch, not_done_batch)
                # get data that matches qtarget and states to corresponding action 
                state_batch = agent.encode_state(state_batch)
                # train agent for 1 step
                agent.train(state_batch, action_batch, batch_size, qtargets)
            
            if agent.training_steps % update_frequency == 0:
                #start_time = time.monotonic()
                print('Updating Q target')  
                agent.updateQTarget()
                print('Updating Q target finished')

                print('Training autoencoder')
                agent.train_autoencoder(memory,1500)
                print('Finished training autoencoder')
            

            if agent.training_steps % evaluation_frequency == 0:
                agent.save_agent(j)
                j+=1

                print('Current epsilon: ',agent.epsilon)
                print('Frames = ', agent.training_steps)
                print('Frames in current session = ', n)
                print('Number of episodes = ', episodes)
                print('Number of saved agents = ',j)
                print('Memory size = ' , len(memory.data))
                print(mem_top())
                 

        episodes += 1

except:
    print('Welp')

#print(mem_top())
file_name = 'FINAL' + str(agent.training_steps)
agent.save_agent(file_name)
np.save('log/idx',j)
print("Total number of frames: ", n)
print("Total number of episodes: ", episodes)
memory.save_replay(file_name)


"""
done = False
initial_state = env.reset()
action = agent.getAction(LazyFrame2Torch(initial_state)) 
state, reward, done, _ = env.step(action)
memory.add(initial_state,action,reward,state,done )
agent.decrease_epsilon()
n += 1

action = agent.getAction(LazyFrame2Torch(state)) 
next_state,reward,done,_ = env.step(action)
memory.add(state,action,reward,next_state,done)
state = next_state
agent.decrease_epsilon()
n += 1

# get batch of size 32 from replay memory
state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch(batch_size)
# get qtargets
qtarget = agent.getQtargets(next_state_batch, reward_batch, not_done_batch)
# get data that matches qtarget and states to corresponding action 
state_batch, qtarget = agent.getTrainingData(state_batch,qtarget,action_batch)
# train agent for 1 step
agent.train(state_batch, qtarget, 1)

agent.updateQTarget()


agent.save_agent(j)
j+=1
print('Current epsilon: ',agent.epsilon)
for i in range(4):
    print('Linear weights ' + str(i) + ' sum: ', np.sum(agent.Linear[i].getWeights()['weight'].detach().numpy()))
print('Frames = ', agent.training_steps)
print('Frames in current session = ', n)
print('Number of episodes = ', episodes)
print('Number of saved agents = ',j)
print('Memory size = ' , len(memory.data))



from collections import defaultdict
from gc import get_objects
before = defaultdict(int)
after = defaultdict(int)
for i in get_objects():
    before[type(i)] += 1 
 
leaked_things = [[x] for x in range(10)]
for i in get_objects():
    #after[type(i)] += 1
    print(i)

if agent.training_steps % 200000 == 0:
    nam = 'backup' + str(agent.training_steps)
    agent.save_agent(nam)
    memory.save_replay(nam)
#end_time = time.monotonic()
#print('Block 4 time: ',timedelta(seconds=end_time - start_time))
"""

"""for i in get_objects():
    after[type(i)] += 1
print([(k, after[k] - before[k]) for k in after if after[k] - before[k]])"""
"""before = defaultdict(int)
after = defaultdict(int)
for i in get_objects():
    before[type(i)] += 1 """