import abc
import numpy as np
import theano
import theano.tensor as T
class Module:
    def __init__(self):
        # save parameters
        self.parameters = {}
        # save gradients
        self.gradients = {}
    def backward(self,loss_func):
        for param in self.parameters:
            self.gradients[param] = T.grad(loss_func,self.parameters[param])
class Linear(Module):
    def __init__(self,in_dim,out_dim,b = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        weight = np.random.rand(in_dim,out_dim)
        self.parameters["weight"] = theano.shared(value=weight,name="weight")
        if b:
            bais = np.random.rand(out_dim)
            self.parameters["bais"] = theano.shared(value=bais,name="bais")
    def forward(self,input):
        weight = self.parameters["weight"]
        if "bais" in self.parameters:
            bais = self.parameters["bais"]
            return T.dot(input,weight) + bais
        else:
            return T.dot(input,weight)
class DoubleLinear(Module):
    def __init__(self,in_dim1,in_dim2,out_dim,b = True):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.out_dim = out_dim

        weightA = np.random.rand(in_dim1,out_dim)
        weightB = np.random.rand(in_dim2,out_dim)
        self.parameters["weightA"] = theano.shared(value=weightA,name="weightA")
        self.parameters["weightB"] = theano.shared(value=weightB,name="weightB")
        if b:
            bais = np.random.rand(out_dim)
            self.parameters["bais"] = theano.shared(value=bais,name="bais")
    def forward(self,input1,input2):
        weightA = self.parameters["weightA"]
        weightB = self.parameters["weightB"]
        if "bais" in self.parameters:
            bais = self.parameters["bais"]
            return T.dot(input1,weightA)+T.dot(input2,weightB)+bais
        else:
            return T.dot(input1,weightA)+T.dot(input2,weightB)
class RNNBase(Module):
    def __init__(self,in_dim,hid_dim,num_layers = 1,bidirectional = False,batch_first = False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
class RNN(RNNBase):
    def __init__(self,in_dim,hid_dim,num_layers=1,bidirectional = False,batch_first = False):
        super().__init__(in_dim,hid_dim,num_layers,bidirectional,batch_first)

        # parameters for RNN
        W_ih = np.random.uniform(-np.sqrt(1.0/in_dim),np.sqrt(1.0/in_dim),(in_dim,hid_dim))
        b_ih = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_hh = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim,hid_dim))
        b_hh = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))

        self.parameters["W_ih"] = theano.shared(value=W_ih,name="W_ih")
        self.parameters['b_ih'] = theano.shared(value=b_ih,name="b_ih")
        self.parameters["W_hh"] = theano.shared(value=W_hh,name="W_hh")
        self.parameters['b_hh']= theano.shared(value=b_hh,name="b_hh")

    def forward(self,input,hid0):
        # forward
        def forward_step(x_t, h_t, W_ih, b_ih, W_hh, b_hh):
            ht = T.tanh(T.dot(x_t, W_ih) + b_ih + T.dot(h_t, W_hh) + b_hh)
            return ht

        output, updates = theano.scan(forward_step,
                                         sequences=input,
                                         outputs_info=[dict(initial=hid0)],
                                         non_sequences=[self.parameters["W_ih"], self.parameters['b_ih'], self.parameters["W_hh"], self.parameters['b_hh']],
                                         strict=True)
        return output,output[-1]
class LSTM(RNNBase):
    def __init__(self,in_dim,hid_dim,num_layers=1,bidirectional = False,batch_first = False):
        super().__init__(in_dim, hid_dim, num_layers, bidirectional, batch_first)

        # parameters for LSTM
        W_ii = np.random.uniform(-np.sqrt(1.0/in_dim),np.sqrt(1.0/in_dim),(in_dim,hid_dim))
        b_ii = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_hi = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim,hid_dim))
        b_hi = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_if = np.random.uniform(-np.sqrt(1.0/in_dim),np.sqrt(1.0/in_dim),(in_dim,hid_dim))
        b_if = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_hf = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim,hid_dim))
        b_hf = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_io = np.random.uniform(-np.sqrt(1.0/in_dim),np.sqrt(1.0/in_dim),(in_dim,hid_dim))
        b_io = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_ho = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim,hid_dim))
        b_ho = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_ig = np.random.uniform(-np.sqrt(1.0/in_dim),np.sqrt(1.0/in_dim),(in_dim,hid_dim))
        b_ig = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))
        W_hg = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim,hid_dim))
        b_hg = np.random.uniform(-np.sqrt(1.0/hid_dim),np.sqrt(1.0/hid_dim),(hid_dim))

        self.parameters["W_ii"] = theano.shared(value=W_ii,name="W_ii")
        self.parameters["b_ii"] = theano.shared(value=b_ii, name="b_ii")
        self.parameters["W_hi"] = theano.shared(value=W_hi, name="W_hi")
        self.parameters["b_hi"] = theano.shared(value=b_hi, name="b_hi")
        self.parameters["W_if"] = theano.shared(value=W_if, name="W_if")
        self.parameters["b_if"] = theano.shared(value=b_if, name="b_if")
        self.parameters["W_hf"] = theano.shared(value=W_hf, name="W_hf")
        self.parameters["b_hf"] = theano.shared(value=b_hf, name="b_hf")
        self.parameters["W_io"] = theano.shared(value=W_io, name="W_io")
        self.parameters["b_io"] = theano.shared(value=b_io, name="b_io")
        self.parameters["W_ho"] = theano.shared(value=W_ho, name="W_ho")
        self.parameters["b_ho"] = theano.shared(value=b_ho, name="b_ho")
        self.parameters["W_ig"] = theano.shared(value=W_ig,name="W_ig")
        self.parameters["b_ig"] = theano.shared(value=b_ig,name="b_ig")
        self.parameters["W_hg"] = theano.shared(value=W_hg,name="W_hg")
        self.parameters["b_hg"] = theano.shared(value=b_hg,name="b_hg")
    def forward(self,input,hid0,chid0):
        # parameters for defination
        W_ii = self.parameters["W_ii"]
        b_ii = self.parameters["b_ii"]
        W_hi = self.parameters["W_hi"]
        b_hi = self.parameters["b_hi"]
        W_if = self.parameters["W_if"]
        b_if = self.parameters["b_if"]
        W_hf = self.parameters["W_hf"]
        b_hf = self.parameters["b_hf"]
        W_io = self.parameters["W_io"]
        b_io = self.parameters["b_io"]
        W_ho = self.parameters["W_ho"]
        b_ho = self.parameters["b_ho"]
        W_ig = self.parameters["W_ig"]
        b_ig = self.parameters["b_ig"]
        W_hg = self.parameters["W_hg"]
        b_hg = self.parameters["b_hg"]
        def forward_step(x_t,h_t,c_t,
                         W_ii,b_ii,W_hi,b_hi,
                         W_if,b_if,W_hf,b_hf,
                         W_ig,b_ig,W_hg,b_hg,
                         W_io,b_io,W_ho,b_ho):
            i_t = T.nnet.sigmoid(T.dot(x_t,W_ii)+b_ii+T.dot(h_t,W_hi)+b_hi)
            f_t = T.nnet.sigmoid(T.dot(x_t,W_if)+b_if+T.dot(h_t,W_hf)+b_hf)
            o_t = T.nnet.sigmoid(T.dot(x_t,W_io)+b_io+T.dot(h_t,W_ho)+b_ho)
            g_t = T.tanh(T.dot(x_t,W_ig)+b_ig+T.dot(h_t,W_hg)+b_hg)
            c_t = f_t*c_t + i_t*g_t
            h_t = o_t*T.tanh(c_t)
            return h_t,c_t
        [output,coutput],updates = theano.scan(forward_step,
                                         sequences=input,
                                         outputs_info=[dict(initial = hid0),dict(initial = chid0)],
                                        non_sequences=[W_ii,b_ii,W_hi,b_hi,
                                                       W_if,b_if,W_hf,b_hf,
                                                       W_ig,b_ig,W_hg,b_hg,
                                                       W_io,b_io,W_ho,b_ho])
        return output,(output[-1],coutput[-1])
class LSTMWithNoForget(RNNBase):
    def __init__(self,in_dim,hid_dim,num_layers=1,bidirectional = False,batch_first = False):
        super().__init__(in_dim, hid_dim, num_layers, bidirectional, batch_first)

        # parameters for LSTM
        W_ii = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_ii = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_hi = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_hi = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_io = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_io = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_ho = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_ho = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_ig = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_ig = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_hg = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_hg = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))

        self.parameters["W_ii"] = theano.shared(value=W_ii, name="W_ii")
        self.parameters["b_ii"] = theano.shared(value=b_ii, name="b_ii")
        self.parameters["W_hi"] = theano.shared(value=W_hi, name="W_hi")
        self.parameters["b_hi"] = theano.shared(value=b_hi, name="b_hi")
        self.parameters["W_io"] = theano.shared(value=W_io, name="W_io")
        self.parameters["b_io"] = theano.shared(value=b_io, name="b_io")
        self.parameters["W_ho"] = theano.shared(value=W_ho, name="W_ho")
        self.parameters["b_ho"] = theano.shared(value=b_ho, name="b_ho")
        self.parameters["W_ig"] = theano.shared(value=W_ig, name="W_ig")
        self.parameters["b_ig"] = theano.shared(value=b_ig, name="b_ig")
        self.parameters["W_hg"] = theano.shared(value=W_hg, name="W_hg")
        self.parameters["b_hg"] = theano.shared(value=b_hg, name="b_hg")
    def forward(self,input,hid0,chid0):
        # parameters for defination
        W_ii = self.parameters["W_ii"]
        b_ii = self.parameters["b_ii"]
        W_hi = self.parameters["W_hi"]
        b_hi = self.parameters["b_hi"]
        W_io = self.parameters["W_io"]
        b_io = self.parameters["b_io"]
        W_ho = self.parameters["W_ho"]
        b_ho = self.parameters["b_ho"]
        W_ig = self.parameters["W_ig"]
        b_ig = self.parameters["b_ig"]
        W_hg = self.parameters["W_hg"]
        b_hg = self.parameters["b_hg"]
        def forward_step(x_t,h_t,c_t,
                         W_ii,b_ii,W_hi,b_hi,
                         W_ig,b_ig,W_hg,b_hg,
                         W_io,b_io,W_ho,b_ho):
            i_t = T.nnet.sigmoid(T.dot(x_t,W_ii)+b_ii+T.dot(h_t,W_hi)+b_hi)
            o_t = T.nnet.sigmoid(T.dot(x_t,W_io)+b_io+T.dot(h_t,W_ho)+b_ho)
            g_t = T.tanh(T.dot(x_t,W_ig)+b_ig+T.dot(h_t,W_hg)+b_hg)
            c_t = c_t + i_t*g_t
            h_t = o_t*T.tanh(c_t)
            return h_t,c_t
        [output, coutput], updates = theano.scan(forward_step,
                                                 sequences=input,
                                                 outputs_info=[dict(initial=(hid0, chid0))],
                                                 non_sequences=[W_ii, b_ii, W_hi, b_hi,
                                                                W_ig, b_ig, W_hg, b_hg,
                                                                W_io, b_io, W_ho, b_ho])
        return output, (output[-1], coutput[-1])
class LSTMPeephole(RNNBase):
    def __init__(self, in_dim, hid_dim, num_layers=1, bidirectional=False, batch_first=False):
        super().__init__(in_dim, hid_dim, num_layers, bidirectional, batch_first)

        # parameters for LSTM
        W_ii = np.random.uniform(-np.sqrt(-1.0 / in_dim), np.sqrt(-1.0 / in_dim), (in_dim, hid_dim))
        b_ii = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        W_hi = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim, hid_dim))
        b_hi = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        V_ci = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim, hid_dim))
        b_ci = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        W_if = np.random.uniform(-np.sqrt(-1.0 / in_dim), np.sqrt(-1.0 / in_dim), (in_dim, hid_dim))
        b_if = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        W_hf = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim, hid_dim))
        b_hf = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        V_cf = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim, hid_dim))
        b_cf = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        W_io = np.random.uniform(-np.sqrt(-1.0 / in_dim), np.sqrt(-1.0 / in_dim), (in_dim, hid_dim))
        b_io = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        W_ho = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim, hid_dim))
        b_ho = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        V_co = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim, hid_dim))
        b_co = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        W_ig = np.random.uniform(-np.sqrt(-1.0 / in_dim), np.sqrt(-1.0 / in_dim), (in_dim, hid_dim))
        b_ig = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))
        W_hg = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim, hid_dim))
        b_hg = np.random.uniform(-np.sqrt(-1.0 / hid_dim), np.sqrt(-1.0 / hid_dim), (hid_dim))

        self.parameters["W_ii"] = theano.shared(value=W_ii, name="W_ii")
        self.parameters["b_ii"] = theano.shared(value=b_ii, name="b_ii")
        self.parameters["W_hi"] = theano.shared(value=W_hi, name="W_hi")
        self.parameters["b_hi"] = theano.shared(value=b_hi, name="b_hi")
        self.parameters["W_if"] = theano.shared(value=W_if, name="W_if")
        self.parameters["b_if"] = theano.shared(value=b_if, name="b_if")
        self.parameters["W_hf"] = theano.shared(value=W_hf, name="W_hf")
        self.parameters["b_hf"] = theano.shared(value=b_hf, name="b_hf")
        self.parameters["W_io"] = theano.shared(value=W_io, name="W_io")
        self.parameters["b_io"] = theano.shared(value=b_io, name="b_io")
        self.parameters["W_ho"] = theano.shared(value=W_ho, name="W_ho")
        self.parameters["b_ho"] = theano.shared(value=b_ho, name="b_ho")
        self.parameters["W_ig"] = theano.shared(value=W_ig, name="W_ig")
        self.parameters["b_ig"] = theano.shared(value=b_ig, name="b_ig")
        self.parameters["W_hg"] = theano.shared(value=W_hg, name="W_hg")
        self.parameters["b_hg"] = theano.shared(value=b_hg, name="b_hg")

        self.parameters["V_ci"] = theano.shared(value=V_ci, name="V_ci")
        self.parameters["b_ci"] = theano.shared(value=b_ci, name="b_ci")
        self.parameters["V_cf"] = theano.shared(value=V_cf, name="V_cf")
        self.parameters["b_cf"] = theano.shared(value=b_cf, name="b_cf")
        self.parameters["V_co"] = theano.shared(value=V_co, name="V_co")
        self.parameters["b_co"] = theano.shared(value=b_co, name="b_co")
    def forward(self,input,hid0,chid0):
        # parameters for defination
        W_ii = self.parameters["W_ii"]
        b_ii = self.parameters["b_ii"]
        W_hi = self.parameters["W_hi"]
        b_hi = self.parameters["b_hi"]
        W_if = self.parameters["W_if"]
        b_if = self.parameters["b_if"]
        W_hf = self.parameters["W_hf"]
        b_hf = self.parameters["b_hf"]
        W_io = self.parameters["W_io"]
        b_io = self.parameters["b_io"]
        W_ho = self.parameters["W_ho"]
        b_ho = self.parameters["b_ho"]
        W_ig = self.parameters["W_ig"]
        b_ig = self.parameters["b_ig"]
        W_hg = self.parameters["W_hg"]
        b_hg = self.parameters["b_hg"]
        V_ci = self.parameters["V_ci"]
        b_ci = self.parameters["b_ci"]
        V_cf = self.parameters["V_cf"]
        b_cf = self.parameters["b_cf"]
        V_co = self.parameters["V_co"]
        b_co = self.parameters["b_co"]
        def forward_step(x_t, h_t, c_t,
                         W_ii, b_ii, W_hi, b_hi,
                         W_if, b_if, W_hf, b_hf,
                         W_ig, b_ig, W_hg, b_hg,
                         W_io, b_io, W_ho, b_ho,
                         V_ci,b_ci,V_cf,b_cf,V_co,b_co):
            i_t = T.nnet.sigmoid(T.dot(x_t, W_ii) + b_ii + T.dot(h_t, W_hi) + b_hi+T.dot(c_t,V_ci)+b_ci)
            f_t = T.nnet.sigmoid(T.dot(x_t, W_if) + b_if + T.dot(h_t, W_hf) + b_hf+T.dot(c_t,V_cf)+b_cf)
            o_t = T.nnet.sigmoid(T.dot(x_t, W_io) + b_io + T.dot(h_t, W_ho) + b_ho+T.dot(c_t,V_co)+b_co)
            g_t = T.tanh(T.dot(x_t, W_ig) + b_ig + T.dot(h_t, W_hg) + b_hg)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * T.tanh(c_t)
            return h_t, c_t

        [output, coutput], updates = theano.scan(forward_step,
                                                 sequences=input,
                                                 outputs_info=[dict(initial=(hid0, chid0))],
                                                 non_sequences=[W_ii,b_ii, W_hi, b_hi,
                                                                W_if,b_if, W_hf, b_hf,
                                                                W_ig,b_ig, W_hg, b_hg,
                                                                W_io,b_io, W_ho, b_ho,
                                                                V_ci,b_ci, V_cf,b_cf,V_co,b_co])
        return output,(output[-1],coutput[-1])
class LSTMCouple(RNNBase):
    def __init__(self,in_dim,hid_dim,num_layers=1,bidirectional = False,batch_first = False):
        super().__init__(in_dim, hid_dim, num_layers, bidirectional, batch_first)
        # parameters for LSTM
        W_ii = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_ii = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_hi = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_hi = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_io = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_io = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_ho = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_ho = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_ig = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_ig = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_hg = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_hg = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))

        self.parameters["W_ii"] = theano.shared(value=W_ii, name="W_ii")
        self.parameters["b_ii"] = theano.shared(value=b_ii, name="b_ii")
        self.parameters["W_hi"] = theano.shared(value=W_hi, name="W_hi")
        self.parameters["b_hi"] = theano.shared(value=b_hi, name="b_hi")
        self.parameters["W_io"] = theano.shared(value=W_io, name="W_io")
        self.parameters["b_io"] = theano.shared(value=b_io, name="b_io")
        self.parameters["W_ho"] = theano.shared(value=W_ho, name="W_ho")
        self.parameters["b_ho"] = theano.shared(value=b_ho, name="b_ho")
        self.parameters["W_ig"] = theano.shared(value=W_ig, name="W_ig")
        self.parameters["b_ig"] = theano.shared(value=b_ig, name="b_ig")
        self.parameters["W_hg"] = theano.shared(value=W_hg, name="W_hg")
        self.parameters["b_hg"] = theano.shared(value=b_hg, name="b_hg")
    def forward(self,input,hid0,chid0):
        # parameters for defination
        W_ii = self.parameters["W_ii"]
        b_ii = self.parameters["b_ii"]
        W_hi = self.parameters["W_hi"]
        b_hi = self.parameters["b_hi"]
        W_io = self.parameters["W_io"]
        b_io = self.parameters["b_io"]
        W_ho = self.parameters["W_ho"]
        b_ho = self.parameters["b_ho"]
        W_ig = self.parameters["W_ig"]
        b_ig = self.parameters["b_ig"]
        W_hg = self.parameters["W_hg"]
        b_hg = self.parameters["b_hg"]
        def forward_step(x_t, h_t, c_t,
                         W_ii, b_ii, W_hi, b_hi,
                         W_ig, b_ig, W_hg, b_hg,
                         W_io, b_io, W_ho, b_ho):
            i_t = T.nnet.sigmoid(T.dot(x_t, W_ii) + b_ii + T.dot(h_t, W_hi) + b_hi)
            o_t = T.nnet.sigmoid(T.dot(x_t, W_io) + b_io + T.dot(h_t, W_ho) + b_ho)
            g_t = T.tanh(T.dot(x_t, W_ig) + b_ig + T.dot(h_t, W_hg) + b_hg)
            c_t = (1-i_t) * c_t + i_t * g_t
            h_t = o_t * T.tanh(c_t)
            return h_t, c_t

        [output, coutput], updates = theano.scan(forward_step,
                                                 sequences=input,
                                                 outputs_info=[dict(initial=(hid0, chid0))],
                                                 non_sequences=[W_ii, b_ii, W_hi, b_hi,
                                                                W_ig, b_ig, W_hg, b_hg,
                                                                W_io, b_io, W_ho, b_ho])
        return output, (output[-1], coutput[-1])
class GRU(RNNBase):
    def __init__(self, in_dim, hid_dim, num_layers=1, bidirectional=False, batch_first=False):
        super().__init__(in_dim, hid_dim, num_layers, bidirectional, batch_first)

        # parameters for GRU
        W_ir = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_ir = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_hr = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_hr = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_iz = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_iz = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_hz = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_hz = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_in = np.random.uniform(-np.sqrt(-1.0/in_dim),np.sqrt(-1.0/in_dim),(in_dim,hid_dim))
        b_in = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))
        W_hn = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim,hid_dim))
        b_hn = np.random.uniform(-np.sqrt(-1.0/hid_dim),np.sqrt(-1.0/hid_dim),(hid_dim))

        self.parameters["W_ir"] = theano.shared(value=W_ir, name="W_ir")
        self.parameters["b_ir"] = theano.shared(value=b_ir, name="b_ir")
        self.parameters["W_hr"] = theano.shared(value=W_hr, name="W_hr")
        self.parameters["b_hr"] = theano.shared(value=b_hr, name="b_hr")
        self.parameters["W_iz"] = theano.shared(value=W_iz, name="W_iz")
        self.parameters["b_iz"] = theano.shared(value=b_iz, name="b_iz")
        self.parameters["W_hz"] = theano.shared(value=W_hz, name="W_hz")
        self.parameters["b_hz"] = theano.shared(value=b_hz, name="b_hz")
        self.parameters["W_in"] = theano.shared(value=W_in, name="W_in")
        self.parameters["b_in"] = theano.shared(value=b_in, name="b_in")
        self.parameters["W_hn"] = theano.shared(value=W_hn, name="W_hn")
        self.parameters["b_hn"] = theano.shared(value=b_hn, name="b_hn")
    def forward(self,input,hid0):
        # parameters for defination
        W_ir = self.parameters["W_ir"]
        b_ir = self.parameters["b_ir"]
        W_hr = self.parameters["W_hr"]
        b_hr = self.parameters["b_hr"]
        W_iz = self.parameters["W_iz"]
        b_iz = self.parameters["b_iz"]
        W_hz = self.parameters["W_hz"]
        b_hz = self.parameters["b_hz"]
        W_in = self.parameters["W_in"]
        b_in = self.parameters["b_in"]
        W_hn = self.parameters["W_hn"]
        b_hn = self.parameters["b_hn"]
        def forward_step(x_t, h_t,
                         W_ir, b_ir, W_hr, b_hr,
                         W_iz, b_iz, W_hz, b_hz,
                         W_in, b_in, W_hn, b_hn):
            r_t = T.nnet.sigmoid(T.dot(x_t, W_ir) + b_ir + T.dot(h_t, W_hr) + b_hr)
            z_t = T.nnet.sigmoid(T.dot(x_t, W_iz) + b_iz + T.dot(h_t, W_hz) + b_hz)
            n_t = T.tanh(T.dot(x_t, W_in) + b_in + r_t*(T.dot(h_t, W_hn) + b_hn))
            h_t = (1-z_t)*n_t + z_t*h_t
            return h_t

        output, updates = theano.scan(forward_step,
                                                 sequences=input,
                                                 outputs_info=[dict(initial=hid0)],
                                                 non_sequences=[W_ir, b_ir, W_hr, b_hr,
                                                                W_iz, b_iz, W_hz, b_hz,
                                                                W_in, b_in, W_hn, b_hn])
        return output, output[-1]