import numpy as np
import time
class Relu:
    def __init__(self):
        self.name = "sigmoid"
    def forward(self,input):
        return np.maximum(input, 0)
    def diff(self,input):
        def inner_diff_relu(x):
            if x >= 0:
                return 1
            else:
                return 0
        inner_diff_relu = np.vectorize(inner_diff_relu)
        return inner_diff_relu(input)
class Sigmoid:
    def __init__(self):
        self.name = "sigmoid"
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        self.sigmoid = sigmoid
    def forward(self,input):
        return 1/(1+np.exp(-input))
    def diff(self,input):
        return np.multiply(self.sigmoid(input),1-self.sigmoid(input))
class Tanh:
    def __init__(self):
        self.name = "tanh"
    def forward(self,input):
        return np.tanh(input)
    def diff(self,input):
        return 1 - np.multiply(np.tanh(input), np.tanh(input))
class Linear:
    def __init__(self,in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        #initialize the weight and bais
        self.weight = np.random.rand(in_dim,out_dim) # size:(in_dim,out_dim)
        self.bais = np.random.rand(out_dim) # size:(1,out_dim)
    def forward(self,input):
        batch = input.shape[0]
        out = np.dot(input,self.weight) + np.broadcast_to(self.bais,(batch,self.out_dim))
        return out
    def __call__(self, input):
        return self.forward(input)
    def __str__(self):
        return "Linear Layer:\n(in_dim=%d,out_dim=%d)"%(self.in_dim,self.out_dim)
class BatchNormalize:
    def __init__(self,dtype = "normalize"):
        self.dtype = dtype
    def forward(self,input):
        if self.dtype == "normalize":
            return BatchNormalize._normalize(input)
        elif self.dtype == "maxmin":
            return BatchNormalize._maxmin(input)
        else:
            raise ValueError("Error for normalization type %s"%str(self.dtype))
    @staticmethod
    def _normalize(input):
        for k in range(len(input)):
            input[k] = (input[k] - input[k].mean())/input[k].std()
        return input
    @staticmethod
    def _maxmin(input):
        for k in range(len(input)):
            input[k] = (input[k] - input[k].min())/(input[k].max()-input[k].min())
        return input
class DNNNet_Regression:
    def __init__(self,dim_list,act_func=None,batchnormalize = False,learning_rate=0.3,lambd = 0,name =None):
        self.dim_list = dim_list
        self.hidden_list = []
        self.num_layers = len(dim_list)-1
        self.batchFlag = batchnormalize
        self.learning_rate = learning_rate
        self.lambd = lambd
        if name is None:
            self.name = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
        else:
            self.name = name
        if act_func is None:
            self.act_class = self.select_act("relu")()
        else:
            self.act_class = self.select_act(act_func)()
        if self.batchFlag:
            self.batch_nomal = BatchNormalize(dtype= "maxmin")
        for k in range(self.num_layers):
            fc = Linear(self.dim_list[k],self.dim_list[k+1])
            self.hidden_list.append(fc)
    def select_act(self,name):
        if name == "sigmoid":
            return Sigmoid
        elif name == "tanh":
            return Tanh
        elif name == "relu":
            return Relu
        else:
            raise TypeError("Unknown type %s"%str(name))
    def forward(self,input):
        out = input
        for k in range(self.num_layers):
            if self.batchFlag:
                out = self.batch_nomal.forward(out)
            hidden = self.act_class.forward(out)
            out = self.hidden_list[k].forward(hidden)
        return out
    def backward(self,input,target):
        weight_grad_list = []
        bais_grad_list = []
        output = self.forward(input)
        delta_l = 2*(output-target)
        for k in range(self.num_layers-1,-1,-1):
            out = input
            for j in range(k):
                if self.batchFlag:
                    out = self.batch_nomal.forward(out)
                hidden = self.act_class.forward(out)
                out = self.hidden_list[j].forward(hidden)
            delta_W = np.dot(out.T,delta_l)
            delta_b = delta_l
            # update the next delta_l
            delta_l = np.multiply(self.act_class.diff(out),np.dot(delta_l,self.hidden_list[k].weight.T))
            # save the gradient
            weight_grad_list.append(delta_W)
            bais_grad_list.append(delta_b)
        return weight_grad_list,bais_grad_list
    def batch_backward(self,input,target):
        batch = input.shape[0]
        for k in range(batch):
            weight_grad_list,bais_grad_list = self.backward(input[k],target[k])
            self.update_grad(weight_grad_list,bais_grad_list)
    def loss(self,output,target):
        out = np.square(target-output)
        return np.sum(out)/len(out)
    def update_grad(self,weight_grad_list,bais_grad_list):
        N = self.dim_list[self.num_layers]
        for k in range(self.num_layers):
            self.hidden_list[k].weight = self.hidden_list[k].weight - self.learning_rate*(weight_grad_list[self.num_layers-k-1]/N+self.lambd*self.hidden_list[k].weight)
            self.hidden_list[k].bais = self.hidden_list[k].bais - self.learning_rate*bais_grad_list[self.num_layers-k-1]/N