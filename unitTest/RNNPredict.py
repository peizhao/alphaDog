import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import math
import torch
from torch import nn
from torch.autograd import Variable
from models.RNNModels import *
from torch import optim

csvFile = '../data/IBM.csv'
CloseIndex = 3

window = 100
hidden_size = 24
batch_size = 8
train_rate = 0.7
input_size = 6
learning_rate = 0.001
num_epoches = 5

def getCSVDataValuesWithLabel(fileName, label):
    """
    Get the CSV file as value and labels
    :param fileName:  the csv file name
    :param label: the column name for the label
    :return: ValueMatrix, label Vector
    """
    dat = pd.read_csv(fileName).dropna()
    X = dat.loc[:, [x for x in dat.columns.tolist() if x != label]].as_matrix()
    Label = dat.loc[:, [x for x in dat.columns.tolist() if x == label]].as_matrix()
    return X, Label

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
    for index in range(data.shape[1]):
        col = data[:,index]
        max_value = np.max(col)
        min_value = np.min(col)
        scalar = max_value - min_value
        data[:,index] = col / scalar

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

def create_UpDownLabel(dataset,look_back=2):
    data, history,labels = [], [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        h = dataset[i:(i + look_back -1), :]
        current = dataset[i + look_back -1]
        previous = dataset[i + look_back -2]
        label=0
        if(current[CloseIndex] > previous[CloseIndex]):
            label = 0
        elif(current[CloseIndex] < previous[CloseIndex]):
            label = 1
        else:
            label = 2
        data.append(a)
        history.append(h)
        labels.append(label)
    return np.array(data), np.array(history),np.array(labels)

def train(model, criterion, optimizer, data, label):
    x = Variable(torch.from_numpy(data).float())
    y = Variable(torch.from_numpy(label).float())
    score = model(x)
    loss = criterion(score, y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def predict(model, data):
    model.eval()
    res = []
    for i in range(len(data)):
        x = data[i,:]
        x = Variable(torch.from_numpy(x).float())
        x = x.unsqueeze(0)
        out = model(x)
        out = out.view(-1)
        res.append(out.data.numpy()[0])
    model.train()
    return res



data = getCSVDataValuesWithoutLabel(csvFile,'Date')
print("Raw data shape: {}".format(data.shape))

Normalize(data)
all, value, label = create_DataAndTarget(data, window)
# for i in range(all.shape[0]):
#     print(all[i,:,CloseIndex])
#     print(value[i,:,CloseIndex])
#     print(label[i])

train_size = int(len(data)*train_rate)
test_size = len(data) - train_size

train_data = value[:train_size, :]
train_label = label[:train_size]
test_data = value[train_size:, :]
test_label = label[train_size:]
# assert(len(train_data) == len(train_label))
# assert(len(test_data) == len(test_label))
print("data: {}, label: {}".format(data.shape, label.shape))
print("train data: {}, train label: {}".format(len(train_data), len(train_label)))
print("test data: {}, test label: {}".format(len(test_data), len(test_label)))

net = LSTMPredict(input_size, hidden_size,2,1)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    perm_idx = np.random.permutation(int(train_size))
    j = 0
    while j < train_size:
        batch_idx = perm_idx[j:(j + batch_size)]
        XValue = np.zeros((len(batch_idx), (window-1),input_size))
        YTarget = np.zeros((len(batch_idx)))

        for k in range(len(batch_idx)):
            XValue[k,:,:] =  train_data[batch_idx[k],:]

        loss = train(net, criterion, optimizer, XValue, YTarget)

        if j % 100*batch_size == 0:
            print("epoch: {}, loss: {}".format(epoch, loss.data[0]))

        j += batch_size

val_Pred = predict(net, value)
plt.figure()
plt.plot(range(0, len(label)), label, 'b', label='real')
plt.plot(range(0, len(val_Pred)), val_Pred, 'r', label='predict')
plt.legend(loc='best')
plt.show()