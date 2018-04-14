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
UseCuda = True

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

def create_DataAndTarget(dataset,look_back=2):
    data, history,target = [], [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        h = dataset[i:(i + look_back -1), :]
        l = dataset[i + look_back -1, CloseIndex]
        data.append(a)
        history.append(h)
        target.append(l)
    return np.array(data), np.array(history),np.array(target)

def create_dataset(dataset, look_back=2):
    all, data, target = [], [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        d = dataset[i:(i + look_back -1)]
        t = dataset[i+look_back -1]
        all.append(a)
        data.append(d)
        target.append(t)
    return np.array(all), np.array(data), np.array(target)

zscore = lambda x : (x - x.mean())/x.std()

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

net = OneDimPredictor(window-1, hidden_size)
loss_function = nn.MSELoss()
if UseCuda:
    net.cuda()
    loss_function.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01)

iter_time = time.time()
for e in range(100):
    var_x = torch.from_numpy(train_data).float()
    var_y = torch.from_numpy(train_target).float()
    var_x = Variable(var_x)
    var_y = Variable(var_y)
    if UseCuda:
        var_x = var_x.cuda()
        var_y = var_y.cuda()
    # forward
    out = net(var_x)
    loss = loss_function(out, var_y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 10 == 0:
        print('Epoch: {}, Loss: {:.5f}, time: {:.3f}'.format(e + 1, loss.data[0], time.time() - iter_time))
        iter_time = time.time()


net = net.eval()
train_data_var = Variable(torch.from_numpy(train_data).float())
test_data_var = Variable(torch.from_numpy(test_data).float())
if UseCuda:
    train_data_var = train_data_var.cuda()
    test_data_var = test_data_var.cuda()
pred_train_target = net(train_data_var)
pred_test_target = net(test_data_var)

if UseCuda:
    pred_train_target = pred_train_target.cpu()
    pred_test_target = pred_test_target.cpu()

pred_train_target = pred_train_target.cpu().view(-1).data.numpy()
pred_test_target = pred_test_target.view(-1).data.numpy()

plt.figure()
plt.plot(range(0, len(raw_data)), raw_data, 'b', label='real')
plt.plot(range(0, train_size), pred_train_target, 'g', label='predictTarget')
plt.plot(range(train_size, train_size+test_size), pred_test_target, 'r', label='predictTest')
plt.legend(loc='best')
plt.show()