import torch
from torch import nn
from torch.autograd import Variable

class OneDimPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(OneDimPredictor, self).__init__()
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

class OneDimClassier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(OneDimClassier, self).__init__()
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