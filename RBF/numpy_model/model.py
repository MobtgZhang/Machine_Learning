import numpy as np
from utils import select_func
from utils import SoftMax
class RBFRegression:
    def __init__(self,in_dim,out_dim,act_name = "Gauss",name =None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act_func = select_func(act_name)()
        self.weight = np.random.rand(in_dim,out_dim)
        self.mu = np.random.rand(in_dim)
        self.sigma = np.random.rand(in_dim)

        self.name = name
    def forward(self,input):
        hidden = self.act_func.forward(input-self.mu,self.sigma)
        return np.dot(hidden,self.weight)
    def backward(self,input,target,learning_rate,lambd):
        batch = input.shape[0]
        output = self.forward(input)
        delta = 2*(target-output)# size of (batch*out_dim)
        R_out = self.act_func.forward(input - self.mu, self.sigma)
        dW = np.dot(R_out.T,delta)/batch
        #Kmeans algorithm
        for k in range(self.in_dim):
            tmp = input[:,k]
            index = 0
            for j in range(batch):
                if np.sum(np.square(tmp - tmp[j])) <np.sum(np.square(tmp - tmp[index])):
                    index = j
            self.mu[k] = tmp[index]
            self.sigma[k] = np.sum(np.square(tmp-self.mu[k]))/batch
        self.weight = self.weight-learning_rate*(dW + lambd*self.weight)
    def loss(self,output,target):
        return np.sum(np.square(output-target))
class RBFClassification:
    def __init__(self,in_dim,n_class,act_name = "Gauss",name = None):
        self.in_dim = in_dim
        self.n_class = n_class

        self.act_func = select_func(act_name)()
        self.weight = np.random.rand(in_dim, n_class)
        self.mu = np.random.rand(in_dim)
        self.sigma = np.random.rand(in_dim)
        self.softmax = SoftMax()

        self.name = name
    def forward(self,input):
        hidden = self.act_func.forward(input-self.mu,self.sigma)
        output = np.dot(hidden,self.weight)
        return self.softmax.forward(output,axis=0)
    def backward(self,input,target,learning_rate,lambd):
        batch = input.shape[0]
        output = self.forward(input)
        delta_r = output - target
        R_out = self.act_func.forward(input - self.mu, self.sigma)
        dW = np.dot(R_out.T, delta_r) / batch
        #Kmeans algorithm
        for k in range(self.in_dim):
            tmp = input[:,k]
            index = 0
            for j in range(batch):
                if np.sum(np.square(tmp - tmp[j])) <np.sum(np.square(tmp - tmp[index])):
                    index = j
            self.mu[k] = tmp[index]
            self.sigma[k] = np.sum(np.square(tmp-self.mu[k]))/batch
        self.weight = self.weight-learning_rate*(dW + lambd*self.weight)
    def loss(self,output,target):
        loss_mat = np.multiply(np.log(output), target)
        length = loss_mat.size
        return -np.sum(loss_mat) / length
class RBFGradRegression:
    def __init__(self,in_dim,out_dim,act_name = "Gauss",name = None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act_func = select_func(act_name)()
        self.weight = np.random.rand(in_dim, out_dim)
        self.mu = np.random.rand(in_dim)
        self.sigma = np.random.rand(in_dim)

        self.name = name
    def forward(self,input):
        hidden = self.act_func.forward(input - self.mu, self.sigma)
        return np.dot(hidden, self.weight)
    def backward(self,input,target,learning_rate,lambd):
        batch = input.shape[0]
        output = self.forward(input)
        delta = 2 * (target - output)  # size of (batch*out_dim)
        R_out = self.act_func.forward(input - self.mu, self.sigma)
        dW = np.dot(R_out.T, delta) / batch

        dsigma = np.multiply(np.multiply(np.divide(2 * np.square(input - self.mu), np.power(self.sigma, 3)),
                                         self.act_func.forward(input - self.mu, self.sigma)),np.dot(delta, self.weight.T)).sum(axis=0) / batch
        dmu = np.multiply(np.multiply(np.divide(2 * (input - self.mu), np.square(self.sigma)),
                        self.act_func.diff(input - self.mu, self.sigma)),np.dot(delta, self.weight.T)).sum(axis=0) / batch





        self.weight = self.weight-learning_rate*(dW + lambd*self.weight)
        self.sigma = self.sigma - learning_rate*dsigma
        self.mu = self.mu - learning_rate*dmu
    def loss(self,output,target):
        return np.sum(np.square(output-target))
class RBFGradClassification:
    def __init__(self,in_dim,n_class,act_name = "Gauss",name = None):
        self.in_dim = in_dim
        self.n_class = n_class
        self.act_func = select_func(act_name)()
        self.weight = np.random.rand(in_dim, n_class)
        self.mu = np.random.rand(in_dim)
        self.sigma = np.random.rand(in_dim)
        self.softmax = SoftMax()

        self.name = name
    def forward(self,input):
        hidden = self.act_func.forward(input - self.mu, self.sigma)
        output = np.dot(hidden, self.weight)
        return self.softmax.forward(output, axis=0)
    def backward(self,input,target,learning_rate,lambd):
        batch = input.shape[0]
        output = self.forward(input)
        delta_r = output - target
        R_out = self.act_func.forward(input - self.mu, self.sigma)
        dW = np.dot(R_out.T, delta_r) / batch
        dsigma = np.multiply(np.multiply(np.divide(2*np.square(input-self.mu),np.power(self.sigma,3)),self.act_func.forward(input-self.mu,self.sigma)),
                             np.dot(delta_r,self.weight.T)).sum(axis=0)/batch
        dmu = np.multiply(
            np.multiply(np.divide(2 * (input - self.mu),np.square(self.sigma)),self.act_func.diff(input - self.mu, self.sigma)),
            np.dot(delta_r, self.weight.T)).sum(axis=0) / batch
        self.weight = self.weight - learning_rate * (dW + lambd * self.weight)
        self.sigma = self.sigma - learning_rate * dsigma
        self.mu = self.mu - learning_rate * dmu
    def loss(self,output,target):
        loss_mat = np.multiply(np.log(output), target)
        length = loss_mat.size
        return -np.sum(loss_mat) / length
class RBFBPRegression:
    def __init__(self,in_dim,hid_dim,out_dim,act_name = "Gauss",name = None):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.act_func = select_func(act_name)()
        self.W_i = np.random.rand(in_dim,hid_dim)
        self.B_i = np.random.rand(hid_dim)
        self.W_o = np.random.rand(hid_dim,out_dim)
        self.B_o = np.random.rand(out_dim)

        self.mu = np.random.rand(hid_dim)
        self.sigma = np.random.rand(hid_dim)

        self.name = name
    def forward(self,input):
        batch = input.shape[0]
        z_out = np.dot(input,self.W_i)+np.broadcast_to(self.B_i,(batch,self.hid_dim))
        hidden = self.act_func.forward(z_out-self.mu,self.sigma)
        return np.dot(hidden,self.W_o)+np.broadcast_to(self.B_o,(batch,self.out_dim))
    def backward(self,input,target,learning_rate,lambd):
        batch = input.shape[0]
        output = self.forward(input)
        delta = 2 * (target - output)  # size of (batch*out_dim)
        hid = np.dot(input, self.W_i) + np.broadcast_to(self.B_i, (batch, self.hid_dim))
        # gradient for dW_o
        dW_o = np.dot(self.act_func.forward(hid - self.mu, self.sigma).T, delta) / batch
        # gradient for dB_o
        dB_o = delta.sum(axis=0) / batch
        # gradient for dW_i
        diff = np.multiply(np.dot(delta,self.W_o.T),self.act_func.diff(hid-self.mu,self.sigma))
        dW_i = np.dot(input.T,diff)/batch
        # gradient for db_i
        dB_i = np.multiply(np.dot(delta,self.W_o.T),self.act_func.diff(hid-self.mu,self.sigma)).sum(axis=0)/batch
        # gradient for dsigma
        R_diff = 2 * np.square(hid - self.mu) / np.power(self.sigma, 3) * self.act_func.forward(hid - self.mu, self.sigma)
        dsigma = np.multiply(np.dot(delta,self.W_o.T),R_diff).sum(axis=0)/batch
        # gradient for dmu
        R_diff = 2 * (hid - self.mu) / np.square(self.sigma) * self.act_func.diff(hid - self.mu, self.sigma)
        dmu = np.multiply(np.dot(delta,self.W_o.T),R_diff).sum(axis=0)/batch
        # update the gradient
        self.W_o = self.W_o - learning_rate*(dW_o + lambd*self.W_o)
        self.B_o = self.B_o - learning_rate*dB_o
        self.W_i = self.W_i - learning_rate*(dW_i + lambd*self.W_i)
        self.B_i = self.B_i - learning_rate*dB_i
        self.sigma = self.sigma - learning_rate*dsigma
        self.mu = self.mu - learning_rate*dmu
    def loss(self,output,target):
        return np.sum(np.square(output-target))
class RBFBPClassification:
    def __init__(self, in_dim, hid_dim, n_class, act_name="Gauss",name = None):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_class = n_class
        self.act_func = select_func(act_name)()
        self.W_i = np.random.rand(in_dim, hid_dim)
        self.B_i = np.random.rand(hid_dim)
        self.W_o = np.random.rand(hid_dim, n_class)
        self.B_o = np.random.rand(n_class)

        self.mu = np.random.rand(hid_dim)
        self.sigma = np.random.rand(hid_dim)
        self.softmax = SoftMax()

        self.name = name
    def forward(self,input):
        batch = input.shape[0]
        z_out = np.dot(input,self.W_i)+np.broadcast_to(self.B_i,(batch,self.hid_dim))
        hidden = self.act_func.forward(z_out-self.mu,self.sigma)
        r_out = np.dot(hidden,self.W_o)+np.broadcast_to(self.B_o,(batch,self.n_class))
        return self.softmax.forward(r_out,axis=0)
    def backward(self,input,target,learning_rate,lambd):
        batch = input.shape[0]
        output = self.forward(input)
        delta = target - output  # size of (batch*out_dim)
        hid = np.dot(input, self.W_i) + np.broadcast_to(self.B_i, (batch, self.hid_dim))
        # gradient for dsigma
        R_diff = np.multiply(np.divide(2 * np.square(hid - self.mu), np.power(self.sigma, 3)),
                             self.act_func.forward(hid - self.mu, self.sigma))
        dsigma = np.multiply(np.dot(delta, self.W_o.T), R_diff).sum(axis=0) / batch
        # gradient for dmu
        R_diff = np.multiply(np.divide(2 * (hid - self.mu), np.square(self.sigma)),
                             self.act_func.diff(hid - self.mu, self.sigma))
        dmu = np.multiply(np.dot(delta, self.W_o.T), R_diff).sum(axis=0) / batch
        # gradient for dW_o
        dW_o = np.dot(self.act_func.forward(hid - self.mu, self.sigma).T, delta) / batch
        # gradient for dB_o
        dB_o = delta.sum(axis=0) / batch
        # gradient for dW_i
        diff = np.multiply(np.dot(delta, self.W_o.T), self.act_func.diff(hid - self.mu, self.sigma))
        dW_i = np.dot(input.T, diff) / batch
        # gradient for db_i
        dB_i = np.multiply(np.dot(delta, self.W_o.T), self.act_func.diff(hid - self.mu, self.sigma)).sum(axis=0) / batch
        '''
        # KMeans

        for k in range(self.in_dim):
            tmp = input[:,k]
            index = 0
            for j in range(batch):
                if np.sum(np.square(tmp - tmp[j])) <np.sum(np.square(tmp - tmp[index])):
                    index = j
            self.mu[k] = tmp[index]
            self.sigma[k] = np.sum(np.square(tmp - self.mu[k])) / batch    
        '''
        # update the gradient
        self.W_o = self.W_o - learning_rate * (dW_o + lambd * self.W_o)
        self.B_o = self.B_o - learning_rate * dB_o
        self.W_i = self.W_i - learning_rate * (dW_i + lambd * self.W_i)
        self.B_i = self.B_i - learning_rate * dB_i
        self.sigma = self.sigma - learning_rate * dsigma
        self.mu = self.mu - learning_rate * dmu
    def loss(self,output,target):
        loss_mat = np.multiply(np.log(output), target)
        length = loss_mat.size
        return -np.sum(loss_mat) / length