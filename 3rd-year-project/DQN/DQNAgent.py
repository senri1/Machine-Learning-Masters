import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle

class Qnetwork(torch.nn.Module):
    def __init__(self,num_actions):
        super(Qnetwork,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0).cuda()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).cuda()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).cuda()
        self.fc1 = nn.Linear(64 * 7 * 7, 512).cuda()
        self.fc2 = nn.Linear(512, num_actions).cuda()
    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        # Resize from (batch_size, 64, 7, 7) to (batch_size,64*7*7)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)




class DQNagent():
    
    def __init__(
        self,
        epsilon=1,
        disc_factor = 0.99,
        num_actions=4
        ):
        self.Q = Qnetwork(num_actions)
        self.Q.cuda()
        self.QTarget = Qnetwork(num_actions).cuda()
        self.QTarget.load_state_dict(self.Q.state_dict())
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.learning_rate = 0.00025
        self.num_actions = num_actions
        self.training_steps = 0
        self.min_epsilon = 0.1
        self.init_epsilon = 1.0
        #optimizer = torch.optim.adam(agent.Qnetwork.parameters(), lr = learning_rate)
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=self.learning_rate, eps=0.01, alpha=0.95)
        self.eps_decay_steps = 1000000.0
    
    def getQvalues(self,state):
        with torch.no_grad():
            q = self.Q(state)
        return q

    def getAction(self,state):
        
        Qvalues = self.getQvalues(state)
        probability = np.random.random_sample()

        if self.epsilon <= probability:
            _, action = Qvalues.max(1)
        else:
            action = np.random.randint(0,high=4)
        return action

    def getQtargets(self, next_state_batch, reward_batch, not_done_batch):
        with torch.no_grad():
            # Get the target q values 
            qtarget, _ = torch.max(self.QTarget(next_state_batch), 1)

            # set final frame in episode to have q value equal to reward
            qtarget = not_done_batch * qtarget

            # calculate target q value r + y * Qt
            qtarget = reward_batch + self.disc_factor * qtarget

        return qtarget

    def train(self, state_batch, action_batch, batch_size, qtargets):
        # Zero any graidents
        self.optimizer.zero_grad()
        # Get the q values corresponding to action taken
        qvalues = self.Q(state_batch)[range(batch_size), action_batch]
        # loss is mean squared loss 
        loss = F.mse_loss(qvalues,qtargets)
        # calculate gradients of q network parameters
        loss.backward()
        # update paramters a single step
        self.optimizer.step()

    def update_target(self):
        self.QTarget.load_state_dict(self.Q.state_dict())

    def decrease_epsilon(self):
        self.training_steps += 1
        self.epsilon = max(self.min_epsilon, self.init_epsilon - (self.init_epsilon-self.min_epsilon) * float(self.training_steps)/self.eps_decay_steps)

    def save_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Directory " , dir ,  " already exists")

        torch.save(self.Q.state_dict(),dir + 'Qnet' + '.pth')
        torch.save(self.QTarget.state_dict(),dir + 'QTargetnet' + '.pth')
        with open(os.getcwd() + '/' + dir + 'metadata.pckl' , "wb") as f:
            pickle.dump([self.epsilon, self.training_steps], f)
        
    def load_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        state_dict = torch.load(dir + 'Qnet' + '.pth')
        self.Q.load_state_dict(state_dict)
        state_dict = torch.load(dir + 'QTargetnet' + '.pth')
        self.QTarget.load_state_dict(state_dict)
        with open(os.getcwd() +'/' + dir +'metadata.pckl', "rb") as f:
            metadata = pickle.load(f)
        self.epsilon = metadata[0]
        self.training_steps = metadata[1]

"""
done = False
initial_state = env.reset()
action = agent.getAction(LazyFrame2Torch(initial_state)) 
state, reward, done, _ = env.step(action)
memory.add(initial_state,action,reward,state,done )

action = agent.getAction(LazyFrame2Torch(state)) 
next_state,reward,done,_ = env.step(action)
memory.add(state,action,reward,next_state,done)
state = next_state
"""
