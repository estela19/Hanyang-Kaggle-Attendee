import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torchvision.datasets import CIFAR10
from torchvision import transforms, models

transformList1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomRotation(90),
        transforms.RandomRotation(315),
        transforms.RandomRotation(270),
        transforms.ColorJitter(30),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transformList2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),    
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform = transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

epoch = 20 + 1

#model = models.resnet18(pretrained = True)
model = models.densenet161()

num_feature = model.classifier.in_features
model.classifier = nn.Linear(num_feature, 10)
net = model.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
#optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, nesterov=True)

loss_func = nn.CrossEntropyLoss()

for i in range(epoch):
  for x, y in data_loader:
    x = x.cuda()
    y = y.cuda()

    pred = net(x)

    loss = loss_func(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print('Epoch %d, acc %0.3f' %(i, torch.mean((pred.argmax(dim=1) == y).float(), dim=-1)))
