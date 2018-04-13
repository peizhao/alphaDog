import torch
from torch import nn
from torch.autograd import Variable
from models.LSTMPred import *
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
def SeriesGen(N):
    x = torch.arange(1,N,0.01)
    return torch.sin(x)

def trainDataGen(seq, k):
    dat = list()
    L = len(seq)
    for i in range(L-k-1):
        indat = seq[i:i+k]
        outdat = seq[i+1:i+k+1]
        dat.append((indat, outdat))
    return dat

def ToVariable(x):
    return Variable(torch.FloatTensor(x))

y = SeriesGen(10)
dat = trainDataGen(y.numpy(), 10)

#model = LSTMPred(1,10)
#model = LSTM2Pred(1,10,1)
model = LSTM_Reg(1,10,1)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    print("epoch {}".format(epoch))
    for index, (seq, outs) in enumerate(dat[:700]):
        seq = ToVariable(seq)
        outs = ToVariable(outs)
        optimizer.zero_grad()
        #model.hidden = model.init_hidden()
        mdout = model(seq)
        loss = loss_function(mdout, outs)
        loss.backward()
        optimizer.step()
        if(index % 100 == 0 ):
            print("index: {} loss is {}".format(index,loss.data.numpy()))

preDat= []
for seq, trueVal in dat[700:]:
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    ret = model(seq)[-1].data.numpy()[0]
    preDat.append(ret)

#print(y)
plt.figure()
plt.plot(y.numpy())
plt.plot(range(700,890), preDat)
plt.show()
