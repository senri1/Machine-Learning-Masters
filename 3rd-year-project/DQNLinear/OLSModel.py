import numpy as np
import torch
import torch.nn.functional as F

class OLS:

    def __init__(self,l2):
        self.dtype = torch.float
        self.device = torch.device("cpu")
        self.Linear = torch.nn.Linear(400,1)
        self.learning_rate = 1e-7
        self.optimizer = torch.optim.SGD(self.Linear.parameters(), lr = self.learning_rate, weight_decay=l2)
    
    def fit(self,X,Y,steps):
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        for t in range(steps):
            self.optimizer.zero_grad()
            y_pred = self.Linear(X)
            loss = F.mse_loss(y_pred,Y)
            loss.backward()
            self.optimizer.step()

    def predict(self,X):
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            X = self.Linear(X)
        return float(X[0])

    def setWeights(self, newWeight):
        self.Linear.load_state_dict(newWeight)

    def getWeights(self):
        return self.Linear.state_dict()
    
    def getSquaredError(self,X,Y):
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        with torch.no_grad():
            y_pred = self.Linear(X)
            squared_error = F.mse_loss(y_pred,Y)
        return float(squared_error)