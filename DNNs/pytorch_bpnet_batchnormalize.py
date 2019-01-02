import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

import os
import shutil

from utils import create_gif,save_gif
class BpDNN_Net(nn.Module):
    def __init__(self,in_dim,hid_dim_list,out_dim,batch_normalize = True):
        super(BpDNN_Net,self).__init__()
        self.in_dim = in_dim
        self.hid_dim_list = hid_dim_list
        self.out_dim = out_dim
        self.number_layers = len(hid_dim_list)-1

        self.batch_normalize = batch_normalize
        self.hidden_layers = []
        self.batch_layers = []
        if self.batch_normalize:
            self.batch_in = nn.BatchNorm1d(self.in_dim,momentum = 0.5)
        # input layer 
        self.input = nn.Linear(in_dim,self.hid_dim_list[0])
        self.init_weight(self.input)
        for k in range(self.number_layers):
            fc = nn.Linear(self.hid_dim_list[k],self.hid_dim_list[k+1])
            self.init_weight(fc)
            setattr(self,"fc%d"%k,fc)
            self.hidden_layers.append(fc)
            if self.batch_normalize:
                bn = nn.BatchNorm1d(self.hid_dim_list[k],momentum = 0.5)
                setattr(self,"bn%d"%k,bn)
                self.batch_layers.append(bn)

		# output layer 
        self.output = nn.Linear(self.hid_dim_list[self.number_layers],self.out_dim)
        self.init_weight(self.output)

    def init_weight(self,layer):
        init.normal(layer.weight,mean = 0,std = 0.1)
        init.constant(layer.bias,0.2)
    def forward(self,x):
        if self.batch_normalize:
            x = self.batch_in(x)
        x = self.input(x)
        x = torch.tanh(x)
        for k in range(self.number_layers):
            if self.batch_normalize:
                x = self.batch_layers[k](x)
            x = self.hidden_layers[k](x)
            x = torch.relu(x)
        out = self.output(x)
        return out
def train():
    x_data = torch.linspace(-1,1,300).unsqueeze(1)
    y_data = 10 * torch.pow(x_data,3) - 6 * x_data + 8
    y_data = y_data/y_data.max()
    tip = 0.03
    y_data = y_data - tip + tip * torch.randn(x_data.size())
    x_data,y_data = Variable(x_data),Variable(y_data)
    in_dim = 1
    hid_dim_list = [10,15,16,20,25]
    out_dim = 1
    file_path = "pytorch_bpnet_batchnormalize"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    bpnet = BpDNN_Net(in_dim,hid_dim_list,out_dim,batch_normalize = True)
    optimizer = optim.SGD(bpnet.parameters(),lr = 0.1)
    loss_func = nn.MSELoss()
    plt.ion()
    for epoch in range(1500):
        optimizer.zero_grad()
        predict = bpnet(x_data)
        loss = loss_func(predict,y_data)
        loss.backward()
        optimizer.step()
        if epoch % 50 ==0:
            plt.cla()
            plt.scatter(x_data.data.numpy(),y_data.data.numpy())
            plt.plot(x_data.data.numpy(),predict.data.numpy(),'-r',lw = 3)
            print(epoch,loss.data.numpy())
            plt.pause(0.1)
            filename = os.path.join(file_path,"pic" + str(epoch//50) + ".png")
            plt.savefig(filename)
    plt.ioff()
    save_gif(file_path,"pytorch_bpnet_batchnormalize.gif")
    shutil.rmtree(file_path,True)
    print ("Directory: " + file_path +" was removed!")
if __name__ == "__main__":
	train()
