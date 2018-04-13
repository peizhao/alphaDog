import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import math
import torch
from torch import nn
from torch.autograd import Variable
from models.LSTMPred import *
from torch import optim

csvFile = '../data/IBM.csv'

def getNormalizeDataFromLabel(fileName, label):
    df = pd.read_csv(fileName).dropna()
    data = df[label].values
    data = data.astype('float32')
    max_value = np.max(data)
    min_value = np.min(data)
    scalar = max_value - min_value
    # data = list(map(lambda x: x / scalar, data))
    data = list(map(lambda x: x, data))
    return data

def create_UpDownLabel(dataset,look_back=2):
    data, labels = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back -1)]
        current = dataset[i + look_back]
        previous = dataset[i + look_back -1]
        label=0
        if(current > previous):
            label = 0
        elif(current < previous):
            label = 1
        else:
            label = 2
        data.append(a)
        labels.append(label)
    return np.array(data), np.array(labels)

def create_UpDownLabel2(dataset,look_back=2):
    data, history,labels = [], [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        h = dataset[i:(i + look_back -1)]
        current = dataset[i + look_back -1]
        previous = dataset[i + look_back -2]
        label=0
        if(current > previous):
            label = 0
        elif(current < previous):
            label = 1
        else:
            label = 2
        data.append(a)
        history.append(h)
        labels.append(label)
    return np.array(data), np.array(history),np.array(labels)

def train(model, criterion, optimizer, data, label):
    x = Variable(torch.from_numpy(data).float())
    y = Variable(torch.from_numpy(label).long())
    score = model(x)
    score = score.view(-1,3)
    loss = criterion(score, y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def val(model, data, label):
    model.eval()
    correct = 0
    for i in range(len(data)):
        x = data[i]
        y = label[i]
        x = Variable(torch.from_numpy(x).float())
        x = x.view(-1,1,1)
        score = model(x)
        score = score.view(-1,3)
        _, pred = score.max(1)
        if(y == pred.data.numpy()):
            correct += 1
    model.train()
    print("correct: {}, total: {}".format(correct, len(data)))
    acc = correct / float(len(data))
    return acc

window = 6
batch_size = 8
train_rate = 0.7
close_data = getNormalizeDataFromLabel(csvFile,'Close')
all ,data, label = create_UpDownLabel2(close_data, look_back=window)

train_size = int(len(data)*train_rate)
test_size = len(data) - train_size

train_data = data[:train_size]
train_label = label[:train_size]
test_data = data[train_size:]
test_label = label[train_size:]
assert(len(train_data) == len(train_label))
assert(len(test_data) == len(test_label))
print("train data: {}, train label: {}".format(len(train_data), len(train_label)))
print("test data: {}, test label: {}".format(len(test_data), len(test_label)))

# for i in range(train_size):
#     data = all[i]
#     label = train_label[i]
#     test = train_data[i]
#     print("last-1: {}, last: {}, label: {}".format(data[window-2], data[window-1], label))
#     print("train {}".format(test))

net = LSTMClassier(1, 8, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(200):
    perm_idx = np.random.permutation(int(train_size))
    j = 0
    while j < train_size:
        batch_idx = perm_idx[j:(j+batch_size)]
        XValue = np.zeros(((window-1),len(batch_idx), 1))
        YTarget = np.zeros((len(batch_idx)))

        for k in range(len(batch_idx)):
            XValue[:,k,0] =  train_data[batch_idx[k]]

        loss = train(net,criterion, optimizer, XValue, YTarget)

        if j % 100*batch_size == 0:
            test_acc = val(net, test_data, test_label)
            train_acc = val(net, train_data, train_label)
            print("epoch: {}, loss: {}, train_acc: {}, test_acc: {}".format(i, loss.data[0], train_acc, test_acc))

        j += batch_size

    # if i+1 % 10 == 0:
    #     acc = val(net,test_data, test_label)
    #     print("epoch: {}, test val acc is : {}".format(i, acc))



