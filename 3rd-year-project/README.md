# 3rd-year-project
Code and stuff for third year project
Contains varients of DQN, mainly modifications to the Q network involving replacing it with an autoencoder + linear model / neural network.
Idea is to try and have perception (autoencoder) and something to make decision based on perception (linear model/neural network).
Specifically the value function can be said to be split into two parts, the convolutional autoencoder and the linear model/neural network. The former is trained using a loss that would help construct a relatively low dimensional latent space representation of the high dimensional state space, for exampled using the mean squared error. The latter is trained using the standard Q learning loss.
