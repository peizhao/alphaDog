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

def create_dataset_with_windows(dataset, look_back=2, window = 1):
    all, data, target = [], [], []
    step = look_back + window
    for i in range(len(dataset) - step):
        a = dataset[i:(i + step)]
        d = dataset[i:(i + look_back)]
        t = dataset[(i+look_back):(i+step)]
        all.append(a)
        data.append(d)
        target.append(t)
    return np.array(all), np.array(data), np.array(target)

def create_testdataset_with_windows(dataset, look_back=2, window = 1):
    all, data, target = [], [], []
    step = look_back + window
    for i in range(0, (len(dataset) - step), step):
        a = dataset[i:(i + step)]
        d = dataset[i:(i + look_back)]
        t = dataset[(i+look_back):(i+step)]
        all.append(a)
        data.append(d)
        target.append(t)
    return np.array(all), np.array(data), np.array(target)

zscore = lambda x : (x - x.mean())/x.std()

window = 30
predict_window = 5
stepWindow = window + predict_window
hidden_size = window*2
batch_size = 8
train_rate = 0.7

raw_data = getCSVDataValuesWithLabel(csvFile,'Close')
raw_data = zscore(raw_data)
all, data, target = create_dataset_with_windows(raw_data, window, predict_window)
# for i in range(all.shape[0]):
#     print(all[i,:])
#     print(data[i,:])
#     print(target[i])
Tall, Tdata, Ttarget = create_testdataset_with_windows(raw_data, window, predict_window)

train_size = int(len(data)*train_rate)
test_size = len(data) - train_size

train_data = data[:train_size,]
train_target = target[:train_size]
test_data = data[train_size:, :]
test_target = target[train_size:]

net = OneDimPredictor(window, hidden_size, predict_window,2)
loss_function = nn.MSELoss()
if UseCuda:
    net.cuda()
    loss_function.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01)

iter_time = time.time()
for e in range(500):
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


# net = net.eval()
# train_data_var = Variable(torch.from_numpy(train_data).float())
# test_data_var = Variable(torch.from_numpy(test_data).float())
# if UseCuda:
#     train_data_var = train_data_var.cuda()
#     test_data_var = test_data_var.cuda()
# pred_train_target = net(train_data_var)
# pred_test_target = net(test_data_var)
# if UseCuda:
#     pred_train_target = pred_train_target.cpu()
#     pred_test_target = pred_test_target.cpu()
# pred_train_target = pred_train_target.cpu().view(-1).data.numpy()
# pred_test_target = pred_test_target.view(-1).data.numpy()

net = net.eval()
p_data = Variable(torch.from_numpy(Tdata).float())
if UseCuda:
    p_data = p_data.cuda()
p_data_target = net(p_data)
if UseCuda:
    p_data_target = p_data_target.cpu()
p_data_target = p_data_target.cpu().view(-1,predict_window).data.numpy()

plt.figure()
for i in range(len(Tall)):
    plt.plot(range(i*stepWindow,(i+1)*stepWindow), Tall[i], 'b')
    plt.plot(range(i*stepWindow+window,(i+1)*stepWindow),p_data_target[i], 'r')
    #plt.plot(range(i * stepWindow + window, (i + 1) * stepWindow), Ttarget[i], 'g')
plt.legend(loc='best')
plt.show()