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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
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

    def forward(self, input):
        output = self.left(input)
        output += self.shortcut(input)
        output = F.relu(output)
        return output

class SE_ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1, ratio = 16):
        super(SE_ResBlock, self).__init__()
        self.outchannel = outchannel
        self.ratio = ratio
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
        self.right = SELayer(outchannel)

    def forward(self, input):
        output = self.left(input)
        output = self.right(output)
        output += self.shortcut(input)
        output = F.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self, block):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(block, 64,  2, stride=1)
        self.layer2 = self.make_layer(block, 128, 2, stride=2)
        self.layer3 = self.make_layer(block, 256, 2, stride=2)
        self.layer4 = self.make_layer(block, 512, 2, stride=2)
        self.fc = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output= self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

transform = torchvision.transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def train(epoch, learning_rate, batch_size, net = None, gpu_available = True, se_available=True):
    writer = SummaryWriter(comment='ResNet')
    device = torch.device("cuda")
    if not gpu_available:
        device = torch.device("cpu")
    if net is None:
        if not se_available:
            net = ResNet(ResBlock)
        else:
            net = ResNet(SE_ResBlock)
        net = net.to(device)
    train = MyData(train=True, transform=transform)
    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    for i in range(epoch):
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
        print("epoch %d: train_acc: %.3f" % (i, correct / 50000))
        writer.add_scalar('Train', total_loss / len(trainset), i)
        test_acc = test(net = net)
        writer.add_scalar('Test', test_acc, i)
        if i % 50 == 0:
            learning_rate = learning_rate / 10
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
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
    parser.add_argument("--learning_rate", dest="lr", default=0.1,type=float)
    parser.add_argument("--gpu", dest="gpu", default=True, type=bool)
    parser.add_argument("--se", dest="se", default=True, type=bool)
    args = parser.parse_args()
    train(epoch=args.epoch, learning_rate=args.lr, batch_size=args.batch_size, gpu_available=args.gpu, se_available=args.se)

