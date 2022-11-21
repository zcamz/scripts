# -*- coding: utf-8 -*-
"""
PyTorch Linear Regression Example
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#generate synthetic training data...
N = 20
seed = 7777
np.random.seed(seed)
X = 10 * np.random.random(N) - 5
Y = -1 + 0.5*X + np.random.randn(N)
n_epochs = 100
losses = np.zeros(n_epochs)
#Plot the training data...
figure, (ax1,ax2) = plt.subplots(2,1)
ax1.scatter(X,Y)
ax1.set_title("Training Data")

#specify model
model = nn.Linear(1,1)
#select loss fcn
criterion = nn.MSELoss()
#choose algorithm
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

#pre-processing and transformations...
#reshape to column vectors
X = X.reshape(N,1)
Y = Y.reshape(N,1)
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

#fit (train) the model...
for it in range(n_epochs):
    optimizer.zero_grad()    
    #forward pass
    outputs = model(inputs)
    loss = criterion(outputs,targets)
    losses[it] = loss.item()    
    #backward pass
    loss.backward()
    optimizer.step()
    print(f'Epoch {it +1}/{n_epochs}, Loss: {loss.item():.4f}')

ax2.plot(losses)
ax2.set_title("Training Loss")
figure.tight_layout()
plt.show()