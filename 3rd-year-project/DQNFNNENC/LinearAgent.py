import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ReplayMemory import ReplayMemory
#from EvaluateAgent import collectRandomData
import pickle
import os


class encoder(torch.nn.Module):
    def __init__(self,num_actions):
        super(encoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0).cuda()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).cuda()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).cuda()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0).cuda()
    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        return out

class decoder(torch.nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.conv1T = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1,padding=0).cuda()
        self.conv2T = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=3, stride=1,padding=0).cuda()
        self.conv3T = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,padding=0).cuda()
        self.conv4T = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=8, stride=4,padding=0).cuda()
        self.conv5T = nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=1, stride=1,padding=0).cuda()

    def forward(self,y):
        out = F.relu(self.conv1T(y))
        out = F.relu(self.conv2T(out))
        out = F.relu(self.conv3T(out))
        out = F.relu(self.conv4T(out))
        out = self.conv5T(out)
        return out

class fnn(torch.nn.Module):
    def __init__(self,num_actions):
        super(fnn,self).__init__()
        self.fc1 = nn.Linear(400, 512).cuda()
        self.fc2 = nn.Linear(512, 256).cuda()
        self.fc3 = nn.Linear(256, 128).cuda()
        self.fc4 = nn.Linear(128, num_actions).cuda()

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return self.fc4(out)


def create_CAE(num_actions, encoder1, decoder1):
    
    print('No encoder, decoder passed in, initializing new ones.')
    encoderObj = encoder(num_actions)
    decoderObj = decoder()
    CAE = nn.Sequential(encoderObj, decoderObj)
    return CAE, encoderObj, decoderObj


    

class Linearagent():
    
    def __init__(
        self,
        epsilon=1,
        disc_factor = 0.99,
        num_actions=4,
        ):
        self.CAE, self.encoder, self.decoder = create_CAE(num_actions, None, None)
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.min_epsilon = 0.05
        self.init_epsilon = 1
        self.eps_decay_steps = 1000000.0
        self.training_steps = 0
        self.num_actions = num_actions
        self.fnn = fnn(num_actions)
        self.fnnTargert = fnn(num_actions)
        self.fnnTargert.load_state_dict(self.fnn.state_dict())
        self.optimizer = torch.optim.Adam(self.CAE.parameters(), lr = 0.001)
        self.fnn_optimizer = torch.optim.Adam(self.fnn.parameters(), lr = 0.0000625, eps = 1.5e-4)


    def train_autoencoder(self, memory, steps):

        for _ in range(steps):
            state_batch,_,_,_,_ = memory.get_batch()
            self.optimizer.zero_grad() 
            loss = F.binary_cross_entropy_with_logits(self.CAE(state_batch),state_batch)
            loss.backward()
            self.optimizer.step()   


    def train(self, state_batch, action_batch, batch_size, qtargets):
        self.training_steps +=1
        # Zero any graidents
        self.fnn_optimizer.zero_grad()
        # Get the q values corresponding to action taken
        qvalues = self.fnn(state_batch)[range(batch_size), action_batch]
        # loss is mean squared loss 
        loss = F.smooth_l1_loss(qvalues,qtargets)
        # calculate gradients of q network parameters
        loss.backward()
        # update paramters a single step
        self.fnn_optimizer.step()


    def getQvalues(self,state):
        with torch.no_grad():
            Q = self.encoder(state)
            Q = self.fnn(Q.view(-1,400))
        return Q

    def getAction(self,state):

        Qvalues = self.getQvalues(state)
        probability = np.random.random_sample()

        if self.epsilon <= probability:
             _, action = Qvalues.max(1)
        else:
            action = np.random.randint(0,high=4)
        del Qvalues
        del probability
        return action

    def encode_state(self,state_batch):

        with torch.no_grad():
            state_batch=self.encoder(state_batch).view(-1,400)
        
        return state_batch

    def getQtargets(self, next_state_batch, reward_batch, not_done_batch):
        with torch.no_grad():
            qtarget = self.encoder(next_state_batch).view(-1,400)
            qtarget = self.fnnTargert(qtarget)
            qtarget, _ = torch.max(qtarget, 1)
            qtarget = not_done_batch * qtarget
            qtarget = reward_batch + self.disc_factor * qtarget

        return qtarget

    def updateQTarget(self):
        self.fnnTargert.load_state_dict(self.fnn.state_dict())

    def decrease_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.init_epsilon - (self.init_epsilon-self.min_epsilon) * float(self.training_steps)/self.eps_decay_steps)


    def save_encoder(self,dir):
        """ saves the CAE as well as the mean and standard deviation of states obtained from the data it was trained on."""
        try:
            torch.save(self.encoder.state_dict(),dir + 'encoder' + '.pth')
            print('saved encoder')
        except:
            print('encoder not saved')
        try:
            torch.save(self.decoder.state_dict(),dir + 'decoder' + '.pth')
            print('saved decoder')
        except:
            print('decoder not saved')

    
    def load_encoder(self,dir):
        """ This method loads a convolutional autoencoder trained on the environment specified in self.env, as well as
            the mean and standard deviation of the states obtained from the training data used to train the CAE. """
        try:
            self.encoder.load_state_dict(torch.load(dir + 'encoder' + '.pth'))
            print('loaded encoder')
        except:
            print('no encoder found will not load.')
        try:
            self.decoder.load_state_dict(torch.load(dir + 'decoder' + '.pth'))
            print('loaded decoder')
        except:
            print('no decoder found will not load.')
        
        self.CAE = nn.Sequential(self.encoder, self.decoder)

        return self.encoder

    def save_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Directory " , dir ,  " already exists")
        try:
            torch.save(self.fnn.state_dict(),dir + 'fnn' + '.pth')
        except:
            print('could not save fnn')
        try:
            torch.save(self.fnnTargert.state_dict(),dir + 'fnnTarget' + '.pth')
        except:
            print('could not save fnn target')
        self.save_encoder(dir)
        try:
            with open(os.getcwd() + '/' + dir + 'metadata.pckl' , "wb") as f:
                pickle.dump([self.epsilon, self.training_steps], f)
        except:
            print('couldnt saved metadata: epsilon and training steps')
        del agent_name
        del dir

    def load_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        try:
            self.fnn.load_state_dict(torch.load(dir + 'fnn' + '.pth'))
            print('loaded fnn')
        except:
            print('could not load fnn')
        try:
            self.fnnTargert.load_state_dict(torch.load(dir + 'fnnTarget' + '.pth'))
            print('loaded target fnn')
        except:
            print('could not load target fnn')
        self.load_encoder(dir)
        try:
            with open(os.getcwd() +'/' + dir +'metadata.pckl', "rb") as f:
                metadata = pickle.load(f)
                self.epsilon = metadata[0]
                self.training_steps = metadata[1]
                print('loaded epsilon and training steps')
        except:
            print('couldnt load metadata which has training steps and current epsilon')
        
