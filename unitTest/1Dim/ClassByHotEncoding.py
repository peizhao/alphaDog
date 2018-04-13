import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import math
import torch
from torch import nn
from torch.autograd import Variable
from OneDimModels import *
from torch import optim
import time

csvFile = '../../data/IBM.csv'
CloseIndex = 3
UseCuda = False

def getCSVDataValuesWithLabel(fileName, label):
    """
    Get the CSV file as value and labels
    :param fileName:  the csv file name
    :param label: the column name for the label
    :return: ValueMatrix, label Vector
    """
    dat = pd.read_csv(fileName).dropna()
    Label = dat.loc[:, [x for x in dat.columns.tolist() if x == label]].as_matrix()
    return Label

def getCSVDataValuesWithoutLabel(fileName, label):
    """
    Get the CSV file as value and labels
    :param fileName:  the csv file name
    :param label: the column name for the label
    :return: ValueMatrix, label Vector
    """
    dat = pd.read_csv(fileName).dropna()
    X = dat.loc[:, [x for x in dat.columns.tolist() if x != label]].as_matrix()
    return X

def getCSVDataValues(fileName):
    dat = pd.read_csv(fileName)
    X = dat.loc[:, [x for x in dat.columns.tolist() ]].as_matrix()
    return X

def Normalize(data):
    for index in range(len(data)):
        col = data[index]
        max_value = np.max(col)
        min_value = np.min(col)
        scalar = max_value - min_value
        data[index] = col / scalar

def one_hot_encode(x, n_classes):
    return np.eye(n_classes)[x]

def test_one_hot_encoding():
    list = [0,1,2,3,4,3,2,1,0]
    n_classes = 5
    one_hot_list = one_hot_encode(list, n_classes)
    print(one_hot_list)

def create_dataset(dataset,look_back=2):
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
        encoding = one_hot_encode(label, 3)
        data.append(a)
        history.append(h)
        labels.append(encoding)
    return np.array(data), np.array(history),np.array(labels)

def val(model ,data, label):
    model.eval()
    var_x = torch.from_numpy(data).float()
    var_y = torch.from_numpy(label).float()
    var_x = Variable(var_x)
    var_y = Variable(var_y)
    if UseCuda:
        var_x = var_x.cuda()
        var_y = var_y.cuda()
    # forward
    out = net(var_x)
    out = out.view(-1, 3)
    _, pred = out.max(1)
    real = label.max(1)
    num_correct = (pred == real).sum().data[0]
    acc = num_correct / float(var_x.shape[0])
    model.train()
    print("correct {}, total {}".format(num_correct, var_x.shape[0]))
    return acc

zscore = lambda x : (x - x.mean())/x.std()
#test_one_hot_encoding()
window = 8
hidden_size = 24
batch_size = 8
train_rate = 0.7

raw_data = getCSVDataValuesWithLabel(csvFile,'Close')
raw_data = zscore(raw_data)
all, data, target = create_dataset(raw_data, window)

# for i in range(all.shape[0]):
#     print(all[i,:])
#     print(data[i,:])
#     print(target[i])

train_size = int(len(data)*train_rate)
test_size = len(data) - train_size

train_data = data[:train_size,]
train_target = target[:train_size]
test_data = data[train_size:, :]
test_target = target[train_size:]

net = OneDimClassier(window-1, hidden_size, 3)
loss_function = nn.MSELoss()
if UseCuda:
    net.cuda()
    loss_function.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01)

iter_time = time.time()
for e in range(2000):
    var_x = torch.from_numpy(train_data).float()
    var_y = torch.from_numpy(train_target).float()
    var_x = Variable(var_x)
    var_y = Variable(var_y)
    if UseCuda:
        var_x = var_x.cuda()
        var_y = var_y.cuda()
    # forward
    out = net(var_x)
    out = out.view(-1, 3)
    loss = loss_function(out, var_y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 10 == 0:
        print('Epoch: {}, Loss: {:.5f}, time: {:.3f}'.format(e + 1, loss.data[0], time.time() - iter_time))
        #val(net, train_data, train_target)
        iter_time = time.time()
