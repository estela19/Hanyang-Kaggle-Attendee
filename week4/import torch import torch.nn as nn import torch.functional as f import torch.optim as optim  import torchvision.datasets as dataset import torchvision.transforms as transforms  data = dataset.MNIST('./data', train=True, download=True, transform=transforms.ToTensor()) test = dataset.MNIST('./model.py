import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim

import torchvision.datasets as dataset
import torchvision.transforms as transforms

data = dataset.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test = dataset.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=32)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=32, drop_last = True)

EPOCH = 0 + 1

class network(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(28, 64, batch_first=True, num_layers=2)
    self.fc = nn.Linear(2*64, 10)

  def forward(self, x):
    hidden = torch.zeros(2, 32, 64)
    output, hidden = self.rnn(x, hidden)
    hidden = torch.reshape(hidden.permute(1,0,2), (32, -1))
    out = self.fc(hidden)
    return out

net = network()

loss_function = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


for i in range(EPOCH):
    for x, y in data_loader:
        x = x.squeeze()
        #x = x.cuda()
        #y = y.cuda()
        #train

        res = net(x)
        loss = loss_function(res, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    inferred_tensor = torch.argmax(res, dim=-1)
    acc = torch.mean((inferred_tensor == y).to(torch.float), dim=-1)
    print('epoch %d acc %.3f'%(i,acc))

#test
net.eval()

for x, y in test_loader:
  x = x.squeeze()

  res = net(x)

inferred_tensor = torch.argmax(res, dim=-1)
acc = torch.mean((inferred_tensor == y).to(torch.float), dim=-1)
print('test acc %.3f' %(acc))
