import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class BpDNN_Net(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,number_layers = 1,batch_normalize = True):
		super(BpDNN_Net,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.number_layers = number_layers
		self.batch_normalize = batch_normalize
		
		self.hidden_layer = []
		self.batch_layer = []

		# input layer 
		self.input = nn.Linear(input_size,hidden_size)
		self.init_weight(self.input)
		if self.batch_normalize:
			self.batch_in = nn.BatchNorm1d(input_size,momentum = 0.5)

		for k in range(self.number_layers):
			fc = nn.Linear(hidden_size,hidden_size)
			self.init_weight(fc)
			setattr(self,"fc%d"%k,fc)
			self.hidden_layer.append(fc)
			if self.batch_normalize:
				bn = nn.BatchNorm1d(hidden_size,momentum = 0.5)
				setattr(self,"bn%d"%k,bn)
				self.batch_layer.append(bn)

		# output layer 
		self.output = nn.Linear(hidden_size,output_size)
		self.init_weight(self.output)

	def init_weight(self,layer):
		init.normal(layer.weight,mean = 0,std = 0.1)
		init.constant(layer.bias,0.2)
	def forward(self,x):
		if self.batch_normalize:
			x = self.batch_in(x)
		x = self.input(x)
		temp = F.tanh(x)
		for k in range(self.number_layers):
			x = self.hidden_layer[k](x)
			if self.batch_normalize:
				x = self.batch_layer[k](x)
			x = F.relu(x)
		out = self.output(x)
		return out
def Main():
	x_data = torch.linspace(-1,1,300).unsqueeze(1)
	y_data = 10 * torch.pow(x_data,3) - 6 * x_data + 8
	#y_data = 10 * torch.pow(x_data,3) - 5 * x_data + 6
	y_data = y_data/y_data.max()
	tip = 0.03
	y_data = y_data -tip + tip * torch.randn(x_data.size())
	x_data,y_data = Variable(x_data),Variable(y_data)

	input_size = 1
	hidden_size = 10
	output_size = 1
	number_layers = 15

	bpnet = BpDNN_Net(input_size,hidden_size,output_size,number_layers,batch_normalize = True)
	optimizer = optim.SGD(bpnet.parameters(),lr = 0.1)
	loss_func = nn.MSELoss()
	plt.ion()
	for epoch in range(15000):
		optimizer.zero_grad()
		predict = bpnet(x_data)
		loss = loss_func(predict,y_data)
		loss.backward()
		optimizer.step()
		print(epoch,loss.data[0])
		if epoch % 50 ==0:
			plt.cla()
			plt.scatter(x_data.data.numpy(),y_data.data.numpy())
			plt.plot(x_data.data.numpy(),predict.data.numpy(),'-r',lw = 3)
			plt.pause(0.000001)
		plt.ioff()
def Main_A():
	# With Learnable Parameters
	m = nn.BatchNorm1d(100)
	# Without Learnable Parameters
	m = nn.BatchNorm1d(100, affine=False)
	input_a = Variable(torch.randn(20, 100))
	output = m(input_a)
	print(input_a.size())
if __name__ == "__main__":
	Main()
