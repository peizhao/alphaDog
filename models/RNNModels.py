import torch
from torch import nn
from torch.autograd import Variable

class LSTMClassier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMClassier, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        # x shoulde be: (batch_size, window_size, input_size)
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

class LSTMPredict(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, out_dim):
        super(LSTMPredict, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x shoulde be: (batch_size, window_size, input_size)
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

class LSTMPredictV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,output_size=1):
        super(LSTMPredictV2, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        # input is seq, window, input_num
        T = seq.data.size()[1]
        res = Variable(seq.data.new(seq.size(0), T, self.input_size).zero_())
        for i in range(T):
            data = seq[:,i,:]
            data = Variable(data.data.unsqueeze(1))
            x,_ = self.rnn(data)
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x
