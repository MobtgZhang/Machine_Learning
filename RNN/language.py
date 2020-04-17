import numpy as np
import theano
import theano.tensor as T
from model import Linear,BiLinear
from model import LSTM,LSTMCouple,LSTMPeephole,LSTMWithNoForget,GRU,RNN
class Vocab:
    def __init__(self):
        self.id2vocab = []
        self.vocab2id = {}
        self.addVocab("<UNK>")
    def addVocab(self,word):
        self.vocab2id.update(dict(word=len(self.id2vocab)))
        self.id2vocab.append(word)
    def __getitem__(self, item):
        if type(item) == str:
            return self.vocab2id[item]
        if type(item) == int:
            return self.id2vocab[item]
class SentSimlairityLSTM:
    def __init__(self,config,vocab_mat =None):
        if vocab_mat is None:
            self.vocab_size = config['vocab_size']
            self.vocab_dim = config['vocab_dim']
            self.embedding = np.random.uniform(-1.0,1.0,(self.vocab_size,self.vocab_dim))
        else:
            self.embedding = vocab_mat
            self.vocab_size = vocab_mat.shape[0]
            self.vocab_dim = vocab_mat.shape[1]
        self.hid_dim = config['hid_dim']
        self.type = config['type']
        if self.type == "LSTM":
            self.lstm = LSTM(self.vocab_dim,self.hid_dim)
        elif self.type == "LSTMCouple":
            self.lstm = LSTMCouple(self.vocab_dim,self.hid_dim)
        elif self.type == "LSTMPeephole":
            self.lstm = LSTMPeephole(self.vocab_dim,self.hid_dim)
        elif self.type == "LSTMWithNoForget":
            self.lstm = LSTMWithNoForget(self.vocab_dim,self.hid_dim)
        else:
            raise TypeError("Error for LSTM model type : %s"%type)
        self.out_dim = config['out_dim']
        self.hidden_flatten = BiLinear(self.hid_dim,self.hid_dim,self.out_dim)
        self.output = Linear(self.out_dim,3)
    def build_model(self):
        sentA = T.tensor3(name="sentenceA")
        sentB = T.tensor3(name="sentenceB")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        hid0 = T.matrix(name="init_hidden")
        chid0 = T.matrix(name="init_chidden")
        # forward output
        _,hidA = self.lstm.forward(sentA,hid0,chid0) # size of (batch,hid_dim)
        _,hidB = self.lstm.forward(sentB,hid0,chid0) # size of (batch,hid_dim)

        h_dot = hidA*hidB
        h_sub = T.abs_(hidA-hidB)

        h_s = T.tanh(self.hidden_flatten.forward(h_dot,h_sub))
        output = T.nnet.softmax(self.output.forward(h_s))
        prediction = T.argmax(output,axis=1)
        # forward
        self.predict = theano.function(inputs=[sentA,sentB,hid0,chid0],
                                       outputs=prediction)
        self._forward = theano.function(inputs=[sentA,sentB,hid0,chid0],
                                       outputs=output)
        # loss function
        loss_func = T.sum(T.nnet.categorical_crossentropy(output,target))/output.shape[0]
        self.loss = theano.function(inputs=[output,target],
                                    outputs=loss_func)
        # backward



    def forward(self,sentA,sentB):
        '''
        :param sentA: size of (batch,seq_len)
        :param sentB: size of (batch,seq_len)
        :return:
        '''