import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

iris = load_iris()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.from_numpy(iris.data).unsqueeze(dim=1).float()
y_train = torch.from_numpy(iris.target).unsqueeze(dim=1).float()

epoch = 1000 + 1

model = nn.Sequential(
    nn.Linear(4, 6),
    nn.ReLU(),
    nn.Linear(6, 10),
    nn.ReLU(),
    nn.Linear(10, 7),
    nn.ReLU(),
    nn.Linear(7, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_train = X_train.squeeze(1)
y_train = y_train.squeeze(1)



for i in range(epoch):
    result = model(X_train)

    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(result, y_train.long())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("x-train")
    print(X_train.shape)
    print(y_train.shape)
    print(result.shape)

    if i % 100 == 0:
        print('epoch %d : Loss %.5f' % (i, loss.item()))
        inferred_tensor = torch.argmax(result, dim=-1)
        acc = torch.mean((inferred_tensor == y_train.long()).to(torch.float), dim=-1)
        print(acc)

