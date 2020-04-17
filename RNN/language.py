import theano
import theano.tensor as T
from model import Module,RNN,GRU,LSTM,LSTMWithNoForget,LSTMPeephole,LSTMCouple
from model import DoubleLinear,Linear
class SentSimlairityLSTM(Module):
    def __init__(self,vocab_dim,hid_dim,out_dim,_type):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        if _type == "LSTM":
            self.lstm = LSTM(self.vocab_dim, self.hid_dim)
        elif _type == "LSTMCouple":
            self.lstm = LSTMCouple(self.vocab_dim, self.hid_dim)
        elif _type == "LSTMPeephole":
            self.lstm = LSTMPeephole(self.vocab_dim, self.hid_dim)
        elif _type == "LSTMWithNoForget":
            self.lstm = LSTMWithNoForget(self.vocab_dim, self.hid_dim)
        else:
            raise TypeError("Error for LSTM model type : %s" % str(_type))

        self.hidden_flatten = DoubleLinear(self.hid_dim, self.hid_dim, self.out_dim)
        self.output = Linear(self.out_dim, 3)
    def forward(self,sentA,sentB,hid0,chid0):
        # forward output
        _, (hidA, _) = self.lstm.forward(sentA, hid0, chid0)  # size of (batch,hid_dim)
        _, (hidB, _) = self.lstm.forward(sentB, hid0, chid0)  # size of (batch,hid_dim)

        h_dot = hidA * hidB
        h_sub = T.abs_(hidA - hidB)

        h_s = T.tanh(self.hidden_flatten.forward(h_dot, h_sub))
        output = T.nnet.softmax(self.output.forward(h_s))
        return output
class SentSimlairityRNNGRU(Module):
    def __init__(self,vocab_dim,hid_dim,out_dim,_type):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        if _type == "RNN":
            self.model = RNN(self.vocab_dim,self.hid_dim)
        elif _type == "GRU":
            self.model = GRU(self.vocab_dim,self.hid_dim)
        else:
            raise TypeError("Error for LSTM model type : %s" % str(_type))
        self.hidden_flatten = DoubleLinear(self.hid_dim, self.hid_dim, self.out_dim)
        self.output = Linear(self.out_dim, 3)
    def forward(self,sentA,sentB,hid0):
        # forward output
        _, hidA = self.model.forward(sentA, hid0)  # size of (batch,hid_dim)
        _, hidB = self.model.forward(sentB, hid0)  # size of (batch,hid_dim)

        h_dot = hidA * hidB
        h_sub = T.abs_(hidA - hidB)

        h_s = T.tanh(self.hidden_flatten.forward(h_dot, h_sub))
        output = T.nnet.softmax(self.output.forward(h_s))
        return output
