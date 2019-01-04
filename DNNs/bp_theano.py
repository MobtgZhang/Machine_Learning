import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
class Linear:
    def __init__(self,in_size,out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.weight = theano.shared(np.random.normal(0,1,(in_size,out_size)))
        self.bais = theano.shared(np.zeros((out_size,)) + 0.1)
        # gradient
        self.gW = None
        self.gb = None
    def forward(self,x):
        W_plus_b = T.dot(x,self.weight) + self.bais
        return W_plus_b
    def update_grad(self,cost):
        self.gW,self.gb = T.grad(cost,[self.weight,self.bais])
        return self.weight,self.gW,self.bais,self.gb
class DNNthE:
    def __init__(self,hid_dim_list):
        self.hid_dim_list = hid_dim_list
        self.num_layers = len(hid_dim_list) - 1
        self.hidden_list = []
        for k in range(self.num_layers):
            fc = Linear(self.hid_dim_list[k],self.hid_dim_list[k+1])
            self.hidden_list.append(fc)
    def forward(self,x):
        out = x
        for k in range(self.num_layers):
            out = self.hidden_list[k].forward(out)
            out = T.tanh(out)
        return out 
    def update_grad(self,cost,learning_rate):
        grad_list = []
        for k in range(self.num_layers):
            weight,gW,bais,gb = self.hidden_list[k].update_grad(cost)
            grad_list.append((weight,weight - learning_rate*gW))
            grad_list.append((bais,bais - learning_rate*gb))
        return grad_list
def Main():
    # make up some data 
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0,0.05,x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    # determine inputs dtype 
    x = T.dmatrix("x")
    y = T.dmatrix("y")
    # defination of the layers 
    output_size = 1
    hid_dim_list =[1,20,1]
    learning_rate = 0.1
    net = DNNthE(hid_dim_list,output_size)
    prediction = net.forward(x)
    # define the cost
    cost = T.mean(T.square(prediction - y))
    # update the grad 
    list_q = net.update_grad(cost,learning_rate)
    # apply gradient descent
    train = theano.function(
        inputs = [x,y],
        outputs = [cost],
        updates = list_q
        )
    # prediction 
    predict = theano.function(inputs = [x],outputs = prediction)
    # training model 
    plt.ion()
    for k in range(5000):
        err = train(x_data,y_data)
        if k %50 == 0:
            y = predict(x_data).reshape(300)
            # x = x_data.reshape(300)
            # print(y.reshape(300).shape,x_data.reshape(300).shape)
            # show data
            plt.cla()
            plt.scatter(x_data,y_data)
            plt.plot(x_data,y,c = 'g',lw = 4)
            plt.pause(0.1)
            print(err[0])
    plt.ioff()
if __name__ == "__main__":
    Main()
