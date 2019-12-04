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
        self.fc2 = nn.Linear(512, num_actions).cuda()

    def forward(self,x):
        out = F.relu(self.fc1(x))
        return self.fc2(out)


def create_CAE(num_actions):
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
        self.CAE, self.encoder, self.decoder = create_CAE(num_actions)
        self.mean = None
        self.sd = None
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.min_epsilon = 0.1
        self.init_epsilon = 1
        self.eps_decay_steps = 1000000.0
        self.training_steps = 0
        self.num_actions = num_actions
        self.fnn = fnn(num_actions)
        self.fnnTargert = fnn(num_actions)
        self.fnnTargert.load_state_dict(self.fnn.state_dict())
        self.optimizer = torch.optim.Adam(self.CAE.parameters(), lr = 0.001)
        self.fnn_optimizer = torch.optim.RMSprop(self.fnn.parameters(), lr=0.00025, eps=0.01, alpha=0.95)

    def train_autoencoder(self, data_size, epochs, batch_size, training_data, env_name):
        if training_data == None:
            training_data = ReplayMemory(data_size, batch_size)
        collectRandomData(training_data, data_size, env_name)
        data = training_data.get_big_batch(data_size)[:,0]
        batch = np.zeros((batch_size,84,84,4), dtype= 'uint8' )
        print(1)
        
        for _ in range(epochs):
            training_loss =0
            np.random.shuffle(data)
            for j in range(int(data_size/batch_size)):
                for n in range(batch_size):
                    batch[n,:,:,:] = data[j*batch_size + n].__array__()
                torch_batch = torch.from_numpy(np.moveaxis(batch,3,1)).float().to('cuda')
                self.optimizer.zero_grad()
                predic = self.CAE(torch_batch)
                loss = F.binary_cross_entropy_with_logits(predic,torch_batch)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    training_loss += float(loss)
            print('Training loss: ',training_loss/(data_size/batch_size) )
        self.mean = self.get_mean(data, batch_size)
        self.sd = self.get_sd(data, batch_size, self.mean)
        del training_data
        del data

    def get_mean(self,data, batch_size):
        data_size = data.size
        mean = 0
        batch = np.zeros((batch_size,84,84,4), dtype= 'uint8' )
        for j in range(int(data_size/batch_size)):
            print(j)
            for n in range(batch_size):
                batch[n,:,:,:] = data[j*batch_size + n].__array__()
                torch_batch = torch.from_numpy(np.moveaxis(batch,3,1)).float().to('cuda')
                with torch.no_grad():
                    mean += np.sum(self.encoder(torch_batch).detach().cpu().numpy().reshape(batch_size,400),axis=0)
        mean = (mean / data_size)[np.newaxis]
        return mean

    def get_sd(self,data, batch_size, mean):
        data_size = data.size
        sd = 0
        batch = np.zeros((batch_size,84,84,4), dtype= 'uint8' )
        for j in range(int(data_size/batch_size)):
            print(j)
            for n in range(batch_size):
                batch[n,:,:,:] = data[j*batch_size + n].__array__()
                torch_batch = torch.from_numpy(np.moveaxis(batch,3,1)).float().to('cuda')
                with torch.no_grad():
                    sd += np.sum((self.encoder(torch_batch).detach().cpu().numpy().reshape(batch_size,400)-mean)**2,axis=0)
        sd = (np.sqrt(sd/data_size))[np.newaxis]
        return sd

    def train(self, state_batch, action_batch, batch_size, qtargets):
        # Zero any graidents
        self.fnn_optimizer.zero_grad()
        # Get the q values corresponding to action taken
        qvalues = self.fnn(state_batch)[range(batch_size), action_batch]
        # loss is mean squared loss 
        loss = F.mse_loss(qvalues,qtargets)
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
        return action

    def encode_state(self,state_batch):

        with torch.no_grad():
            state_batch=self.encoder(state_batch)
            state_batch = state_batch.view(-1,400).detach().cpu().numpy()
            state_batch = state_batch - self.mean
            state_batch = np.divide(state_batch, self.sd, out=np.zeros_like(state_batch), where=self.sd!=0)
            state_batch = torch.from_numpy(state_batch).float().to('cuda')
        
        return state_batch

    def getQtargets(self, next_state_batch, reward_batch, not_done_batch):
        with torch.no_grad():
            next_state_batch = self.encoder(next_state_batch)
            next_state_batch = next_state_batch.view(-1,400).detach().cpu().numpy()
            next_state_batch = next_state_batch - self.mean
            next_state_batch = np.divide(next_state_batch, self.sd, out=np.zeros_like(next_state_batch), where=self.sd!=0)
            qtarget = self.fnnTargert(torch.from_numpy(next_state_batch).float().to('cuda'))
            qtarget, _ = torch.max(qtarget, 1)
            qtarget = not_done_batch * qtarget
            qtarget = reward_batch + self.disc_factor * qtarget

        return qtarget

    def updateQTarget(self):
        self.fnnTargert.load_state_dict(self.fnn.state_dict())

    def decrease_epsilon(self):
        self.training_steps +=1
        self.epsilon = max(self.min_epsilon, self.init_epsilon - (self.init_epsilon-self.min_epsilon) * float(self.training_steps)/self.eps_decay_steps)


    def save_encoder(self,dir):
        """ saves the CAE as well as the mean and standard deviation of states obtained from the data it was trained on."""
        try:
            torch.save(self.encoder.state_dict(),dir + 'encoder' + '.pth')
        except:
            print('encoder not saved')
        try:
            torch.save(self.decoder.state_dict(),dir + 'decoder' + '.pth')
        except:
            print('decoder not saved')
        try:
            torch.save(self.CAE.state_dict(),dir + 'CAE' + '.pth')
        except:
            print('CAE not saved')
        np.save(os.getcwd() + '/' + dir + 'mean' ,self.mean)
        np.save(os.getcwd() + '/' + dir + 'sd',self.sd)
    
    def load_encoder(self,dir):
        """ This method loads a convolutional autoencoder trained on the environment specified in self.env, as well as
            the mean and standard deviation of the states obtained from the training data used to train the CAE. """
        try:
            state_dict = torch.load(dir + 'encoder' + '.pth')
            self.encoder.load_state_dict(state_dict)
        except:
            print('no encoder found will not load.')
        try:
            state_dict = torch.load(dir + 'decoder' + '.pth')
            self.decoder.load_state_dict(state_dict)
        except:
            print('no decoder found will not load.')
        try:
            state_dict = torch.load(dir + 'CAE' + '.pth')
            self.CAE.load_state_dict(state_dict)
        except:
            print('no CAE found will not load.')
        self.mean = np.load(os.getcwd() + '/' + dir + 'mean' + '.npy')
        self.sd = np.load(os.getcwd() + '/' + dir + 'sd' + '.npy')
        return self.encoder

    def save_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Directory " , dir ,  " already exists")

        torch.save(self.fnn.state_dict(),dir + 'fnn' + '.pth')
        torch.save(self.fnnTargert.state_dict(),dir + 'fnnTarget' + '.pth')
        self.save_encoder(dir)
        with open(os.getcwd() + '/' + dir + 'metadata.pckl' , "wb") as f:
            pickle.dump([self.epsilon, self.training_steps], f)

    def load_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        
        state_dict = torch.load(dir + 'fnn' + '.pth')
        self.fnn.load_state_dict(state_dict)
        state_dict = torch.load(dir + 'fnnTarget' + '.pth')
        self.fnnTargert.load_state_dict(state_dict)
        self.load_encoder(dir)
        with open(os.getcwd() +'/' + dir +'metadata.pckl', "rb") as f:
            metadata = pickle.load(f)
        self.epsilon = metadata[0]
        self.training_steps = metadata[1]
