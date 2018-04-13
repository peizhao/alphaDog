import torch
from torch import nn
from torch.autograd import Variable

class LSTMPred(nn.Module):
    def __init__(self, input_size, hidden_size, output_size =1):
        super(LSTMPred,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        weight = Variable(torch.zeros(1,1,self.hidden_size))
        cell = Variable(torch.zeros(1,1,self.hidden_size))
        return (weight, cell)

    def forward(self, seq):
        seq = seq.view(len(seq), 1, -1)
        lstm_out, self.hidden = self.lstm(seq, self.hidden)
        out = self.hidden2out(lstm_out.view(len(seq), -1))
        return out

class LSTM2Pred(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM2Pred, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        outputs = []
        h_t = Variable(torch.zeros(seq.size(0), self.hidden_size), requires_grad=False)
        c_t = Variable(torch.zeros(seq.size(0), self.hidden_size), requires_grad=False)
        seq = seq.view(len(seq), 1, -1)
        lstm_out, hidden = self.lstm(seq, (h_t,c_t))
        out = self.linear(lstm_out.view(len(seq), -1))
        return out

class LSTM_Reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTM_Reg, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        seq = seq.view(len(seq), 1, -1)
        x, _ = self.rnn(seq) # (seq, batch, hidden)
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x

class LSTMClassier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTMClassier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        # input data (seq_len, batch, input_size)
        x, (hidden, cell) = self.lstm(data)
        s,b,h = hidden.shape
        y = hidden.view(s*b,h)
        y = self.reg(y)
        y = y.view(s,b,-1)
        return y  # return (seq_len, batch, input_size)

if __name__ == "__main__":
    x = torch.randn(9,8,1)
    x = Variable(x)
    net = LSTMClassier(1,16,3)
    y = net(x)
    print(y.shape)