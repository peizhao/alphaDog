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
    data = list(map(lambda x: x / scalar, data))
    return data

def create_dataset(dataset, look_back=2):
    data, target = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        data.append(a)
        target.append(dataset[i + look_back])
    return np.array(data), np.array(target)

def create_UpDownLabel(dataset,look_back=2):
    data, target = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        data.append(a)
        current = dataset[i + look_back]
        previous = dataset[i + look_back -1]
        label=0
        if(current > previous):
            label = 0
        elif(current < previous):
            label = 1
        else:
            label = 2
        target.append(label)
    return np.array(data), np.array(target)


# df = pd.read_csv(csvFile).dropna()
# CloseData = df['Close'].values
# CloseData = CloseData.astype('float32')
# max_value = np.max(CloseData)
# min_value = np.min(CloseData)
# scalar = max_value - min_value
# CloseData = list(map(lambda x: x/scalar, CloseData))

window_size =10
rawdata = getNormalizeDataFromLabel(csvFile,'Close')
data, target = create_dataset(rawdata, look_back=window_size)
trainRate = 0.7
train_size = int(len(data)*0.7)
test_size = len(data) - train_size
# setup the train and test data
train_data = data[:train_size]
train_target = target[:train_size]
test_data = data[train_size:]
test_target = data[train_size:]

train_data = train_data.reshape(-1,1,window_size)
train_target = train_target.reshape(-1,1,1)
test_data = test_data.reshape(-1,1,window_size)

train_data = torch.from_numpy(train_data)
train_target = torch.from_numpy(train_target)
test_data = torch.from_numpy(test_data)

# set up the model
net = LSTM_Reg(window_size, 36)
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for e in range(2000):
    var_x = Variable(train_data)
    var_y = Variable(train_target)
    # forward
    out = net(var_x)
    loss = loss_function(out, var_y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 100 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

net = net.eval()
pred_train_target = net(Variable(train_data))
pred_train_target = pred_train_target.view(-1).data.numpy()
pred_test_target = net(Variable(test_data))
pred_test_target = pred_test_target.view(-1).data.numpy()

plt.figure()
plt.plot(range(0, len(rawdata)), rawdata, 'b', label='real')
plt.plot(range(0, train_size), pred_train_target, 'g', label='predictTarget')
plt.plot(range(window_size + train_size, len(rawdata)),pred_test_target,'r', label='predictTest')
plt.legend(loc='best')
plt.show()