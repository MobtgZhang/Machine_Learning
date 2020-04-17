import numpy as np
import theano
import theano.tensor as T

from language import SentSimlairityLSTM,SentSimlairityRNNGRU
import optim
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
class Simlarity:
    def __init__(self,config,vocab_mat = None):
        if vocab_mat is None:
            self.vocab_size = config["vocab_size"]
            self.vocab_dim = config["vocab_dim"]
            self.embeddings = np.array(self.vocab_size,self.vocab_dim)
        else:
            self.vocab_size = vocab_mat.shape[0]
            self.vocab_dim = vocab_mat.shape[1]
            self.embeddings = vocab_mat
        self.config = config
        self.build_model()
    def build_model(self,):
        sentA = T.tensor3(name="sentenceA")
        sentB = T.tensor3(name="sentenceB")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        self.optimizer = optim.SGD(learning_rate,lambd)
        if "LSTM" in self.config["type"]:
            hid0 = T.matrix(name="init_hidden")
            chid0 = T.matrix(name="init_chidden")
            self.model = SentSimlairityLSTM(self.vocab_dim,self.config["hid_dim"],self.config["out_dim"],self.config["type"])
            inputs = [sentA,sentB,hid0,chid0]

        elif ("GRU" in self.config["type"]) or ("RNN" in self.config["type"]):
            hid0 = T.matrix(name="init_hidden")
            self.model = SentSimlairityRNNGRU(self.vocab_dim,self.config["hid_dim"],self.config["out_dim"],self.config["type"])
            inputs = [sentA,sentB,hid0]
        else:
            raise TypeError("Unknown model %s"%self.config["type"])

        output = self.model.forward(*inputs)
        prediction = T.argmax(output, axis=1)
        # forward
        self._predict = theano.function(inputs=inputs,
                                       outputs=prediction)
        self._forward = theano.function(inputs=inputs,
                                        outputs=output)
        # loss function
        loss_func = T.sum(T.nnet.categorical_crossentropy(output, target)) / output.shape[0]
        self.loss = theano.function(inputs=[output, target], outputs=loss_func)
        self.model.backward(loss_func)
        self.optimizer.initial(inputs, self.model)
    def changeIndexToVec(self,sentence):
        exit()
    def forward(self,sentA,sentB):
        self._forward_predict(sentA,sentB,True)
    def predict(self,sentA,sentB):
        self.forward(sentA,sentB,False)
    def _forward_predict(self,sentA,sentB,_type):
        batch = sentA[0]
        sentA = self.changeIndexToVec(sentA)
        sentB = self.changeIndexToVec(sentB)
        if _type:
            func = self._forward
        else:
            func = self._predict
        if "LSTM" in self.config["type"]:
            hid0 = np.random.rand(batch,self.config["hid_dim"])
            chid0 = np.random.rand(batch,self.config["hid_dim"])
            return func(sentA,sentB,hid0,chid0)
        elif ("GRU" in self.config["type"]) or ("RNN" in self.config["type"]):
            hid0 = np.random.rand(batch,self.config["hid_dim"])
            return func(sentA,sentB,hid0)
        else:
            raise TypeError("Unknown model %s" % self.config["type"])