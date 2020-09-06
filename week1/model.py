import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv("weight-height.csv")

X, y = data['Height'], data['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.from_numpy(data['Height'].values).unsqueeze(dim=1).float()
y_train = torch.from_numpy(data['Weight'].values).unsqueeze(dim=1).float()

epoch = 4000 + 1

model = nn.Linear(1, 1)

optimizer = optim.Adam(model.parameters(), lr=1e-1)


for i in range(epoch):
    result = model(X_train)

    loss_func = nn.MSELoss()
    loss = loss_func(result, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print('epoch %d : Loss %.5f' % (i, loss.item()))

