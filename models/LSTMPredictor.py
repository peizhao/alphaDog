import torch
from torch import nn
from torch.autograd import Variable

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTMPredictor, self).__init__()
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

class LSTMPredictorWithHidden(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTMPredictorWithHidden, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        # input data (seq_len, batch, input_size)
        x, (hidden, cell) = self.rnn(seq)
        s,b,h = hidden.shape
        y = hidden.view(s*b,h)
        y = self.reg(y)
        y = y.view(s,b,-1)
        return y  # return (seq_len, batch, input_size)

if __name__ == "__main__":
    x = torch.randn(9, 8, 5)
    x = Variable(x)
    net = LSTMPredictorWithHidden(5, 16, 1)
    y = net(x)
    print(y.shape)