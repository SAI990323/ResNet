import argparse

import load
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter


class MyData(data.Dataset):
    def __init__(self, train=True, transform=None, transform_target=None):
        self.train = train
        if self.train:
            self.data, self.target = load.train()
        else:
            self.data, self.target = load.test()
        self.target = torch.LongTensor(self.target)
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.transform_target is not None:
            target = self.transform_target(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def train(epoch, learning_rate, batch_size, net = None, gpu_available = True):
    writer = SummaryWriter(comment='Alexnet')
    device = torch.device("cuda")
    if not gpu_available:
        device = torch.device("cpu")
    if net is None:
        net = ResNet(ResidualBlock)
        net = net.to(device)
    train = MyData(train=True, transform=transform)
    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    last_correct = 0
    for i in range(epoch):
        if i == 30 and learning_rate > 0.002:
           learning_rate = learning_rate / 10
           optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        total_loss = 0
        correct = 0
        for data, target in trainset:
            inputs = data.to(device)
            targets = target.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = lossfunc(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == targets).sum().item()
        writer.add_scalar('Train', total_loss / len(trainset), i)
        test_acc = test(net = net)
        writer.add_scalar('Test', test_acc, i)
        if abs(correct - last_correct) < 10:
            learning_rate = learning_rate / 10
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        if learning_rate < 0.00001:
            break
        last_correct = correct
    writer.close()

def test(batch_size=128, net = None):
    if net is None:
        net = torch.load('Alexnet.model')
    device = torch.device("cuda")
    test = MyData(train=False, transform=transform)
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testset:
            inputs = data.to(device)
            targets = target.to(device)
            outputs = net(inputs)
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total = total + len(inputs)
    print('test data accuracy: ', correct / total)
    return correct / total



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", dest="batch_size", default=128, type=int)
    parser.add_argument("--epoch", dest="epoch", default=40, type=int)
    parser.add_argument("--learning_rate", dest="lr", default=0.01,type=float)
    parser.add_argument("--gpu", dest="gpu", default=True, type=bool)
    args = parser.parse_args()
    train(epoch=args.epoch, learning_rate=args.lr, batch_size=args.batch_size, gpu_available=args.gpu)

