import torch
import torch.nn as nn
import os
import tongyq24Files.code.Modeling_Framework.DNN as DNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((96,96)),
#    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)


def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            output = net(X)
            predicted = output.argmax(dim=1)
            correct += (predicted == y).sum().item()  
            total += y.size(0) 
    return correct/total

def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('train on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        num = 0
        sum_loss = 0
        accuracy = 0
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                sum_loss += l * X.shape[0]
                num += X.shape[0]
                accuracy += (y_hat.argmax(dim=1) == y).sum().item()
        test_acc = evaluate_accuracy(net, data_iter=test_iter)
        print(f"loss: {(sum_loss / num):.3f}, train_acc: {(accuracy/num):.3f}, test_acc: {test_acc}:.3f")

if __name__ == '__main__':
    lr, num_epochs = 0.001, 10
    net = DNN.RegNetX32()
    train(net, train_loader, test_loader, num_epochs, lr, 'cuda:0')

        