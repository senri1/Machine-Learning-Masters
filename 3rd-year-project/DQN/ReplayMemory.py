from collections import deque
import random
import numpy as np
import torch 
import os 
import pickle
import time

class ReplayMemory():
    def __init__(self, size, batch_size):
        self.max_size = size
        self.current_size = 0
        self.data = deque(maxlen=size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        if self.current_size < self.max_size:
            self.data.append((state, int(action), reward, next_state, not done))
            self.current_size += 1
        else: 
            self.data.popleft()
            self.data.append((state, int(action), reward, next_state, not done))
    
    def get_batch(self):
        batch = random.sample(self.data, self.batch_size)
        batch = np.array(batch)
        statenp = np.zeros((self.batch_size,4,84,84))
        next_statenp = np.zeros((self.batch_size,4,84,84))

        for n in range(batch.shape[0]):
            statenp[n,:,:,:] = (np.moveaxis(batch[n,0].__array__()[np.newaxis,:,:,:],3,1)).astype(float)
            next_statenp[n,:,:,:] = (np.moveaxis(batch[n,3].__array__()[np.newaxis,:,:,:],3,1)).astype(float)

        state = torch.from_numpy(statenp).float().to('cuda')
        action = torch.from_numpy(batch[:,1].astype(float)).long().to('cuda')
        reward = torch.from_numpy(batch[:,2].astype(float)).float().to('cuda')
        next_state = torch.from_numpy(next_statenp).float().to('cuda')
        not_done = torch.from_numpy(batch[:,4].astype(float)).float().to('cuda')
        
        return state, action, reward, next_state, not_done
        
    def save_replay(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Directory " , dir ,  " already exists")
        np.save(dir + 'max_size',self.max_size)
        np.save(dir + 'current_size',self.current_size)
        print(str(self.current_size/500))
        for i in range(int(self.current_size/500)):
            print('From ' + str(i*500) + 'to' + str((i+1)*500) )
            a = np.array(list(self.data)[i*500:(i+1)*500])
            print(a.shape)
            with open(os.getcwd() +'/'+ dir + "mem" + str(i), 'wb') as pfile:
                pickle.dump(a, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            time.sleep(0.1)


    def load_replay(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        self.max_size = int(np.load(dir + 'max_size.npy'))
        self.current_size = np.load(dir + 'current_size.npy')
        self.data = deque(maxlen=self.max_size)
        for i in range(int(self.current_size/500)):
            print(i)
            with open(os.getcwd() + '/' + dir + "mem" + str(i), "rb") as f:
                a = pickle.load(f)
            for j in range(a.shape[0]):
                self.data.append(a[j,:])
            time.sleep(0.1)

