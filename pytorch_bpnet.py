import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BPNetWork(nn.Module):
	def __init__(self,in_size,hidden_size,out_size):
		super(BPNetWork,self).__init__()
		self.hidden = nn.Linear(in_size,hidden_size)
		self.output = nn.Linear(hidden_size,out_size)
	def forward(self,inputq):
		temp = F.relu6(self.hidden(inputq))
		temp = self.output(temp)
		return temp
def Main():
	x_data = torch.linspace(-1,1,300).unsqueeze(1)
	y_data = 10 * torch.pow(x_data,3) - 6 * x_data + 8
	#y_data = 10 * torch.pow(x_data,3) - 5 * x_data + 6
	y_data = y_data/y_data.max()
	tip = 0.03
	y_data = y_data -tip + tip * torch.randn(x_data.size())
	x_data,y_data = Variable(x_data),Variable(y_data)
	in_size = 1
	hidden_size  = 1500
	out_size = 1
	bpnet = BPNetWork(in_size,hidden_size,out_size)
	optimizer = optim.SGD(bpnet.parameters(),lr = 0.005)
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
if __name__ == "__main__":
	Main()
