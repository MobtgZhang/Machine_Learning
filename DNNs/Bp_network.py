import numpy as np
import utils
import os

import shutil
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration/(2  * stepsize))
    x = np.abs(iteration/stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
    return lr
class Linear:
	def __init__(self,in_dim,out_dim):
		self.in_dim = in_dim
		self.out_dim = out_dim
		# initalize the weight and bais in_dim*out_dim
		self.weight =  np.random.rand(in_dim,out_dim)
		self.bais = np.random.rand(out_dim)
	def forward(self,x):
		# input the value length*in_dim
		y = np.matmul(x,self.weight)+self.bais
		return y
	def __str__(self):
		return "Linear layer: \nin_dim: "+str(self.in_dim)+"," + "out_dim: "+str(self.out_dim)
class BatchNormalize:
	def __init__(self):
		self.mean = 0
		self.std = 0
	def forward(self,x):
		self.mean = x.mean()
		self.std = x.std()
		return (x - self.mean)/self.std
class DNNNet:
	def __init__(self,hid_dim_list):
		self.hid_dim_list = hid_dim_list
		self.hidden_list = []
		self.num_layers = len(hid_dim_list)-1
		self.nums = hid_dim_list[self.num_layers]
		for k in range(self.num_layers):
			fc = Linear(self.hid_dim_list[k],self.hid_dim_list[k+1])
			self.hidden_list.append(fc)
		self.batch = BatchNormalize()
	def forward(self,x):
		for k in range(self.num_layers):
			activate = utils.sigmoid(x)
			x = self.hidden_list[k].forward(activate)
			# x = self.batch.forward(x)
		activate = utils.sigmoid(x)
		return activate
	def backward(self,x,y_true):
		bais_list = []
		weight_list = []
		for k in range(self.num_layers,0,-1):
			# calculdate delta
			if k==self.num_layers:
				out = x
				for j in range(k):
					activate = utils.sigmoid(out)
					out = self.hidden_list[j].forward(activate)
				delta = utils.diff_sigmoid(out)*(utils.sigmoid(out)-y_true) # 9*7
			else:
				# update the delta
				weight_post = self.hidden_list[k].weight
				delta_post = bais_list[self.num_layers - k-1]
				out = x
				for j in range(k):
					activate = utils.sigmoid(out)
					out = self.hidden_list[j].forward(activate)
				delta = utils.diff_sigmoid(out)*np.matmul(delta_post,weight_post.T)
			# print(delta.shape)
			# calculdate weight
			out = x
			for j in range(k-1):
				activate = utils.sigmoid(out)
				out = self.hidden_list[j].forward(activate)
			activate = utils.sigmoid(out)
			weight_grad = np.matmul(activate.T,delta) # 10*7

			# add bais_list and weight_list
			bais_list.append(delta)
			weight_list.append(weight_grad)
		return weight_list,bais_list
	def update_grad_sgd(self,weight_list,bais_list,learning_rate = 1,lamda = 0):
		N = self.nums
		# SGD algorithm
		for k in range(self.num_layers):
			self.hidden_list[k].weight = self.hidden_list[k].weight - weight_list[self.num_layers - k-1]/N + lamda * self.hidden_list[k].weight
			self.hidden_list[k].bais = self.hidden_list[k].bais - bais_list[self.num_layers - k-1]/N
	'''
	def update_grad_sgdr(self,weight_list,bais_list,lamda = 0):
		N = self.nums
		# SGDR algorithm
		lrMax = 1
		lrMin = 0

		learning_rate = lrMin + 0.5*(lrMax - lrMin)(1 + )
		for k in range(self.num_layers):
			pass
	'''
	def loss(self,x,y_true):
		y_pred = self.forward(x)
		return utils.quadlf(y_pred,y_true)
	def __str__(self):
		line = ""
		for k in range(self.num_layers):
			line = line + "("+str(k)+")" +"Linear layer: \nin_dim: "+str(self.hidden_list[k].in_dim)+"," + "out_dim: "+str(self.hidden_list[k].out_dim) +"\n"
		return line
def main():
    x = np.linspace(-1,1,300)
    y = 10 * np.power(x,3) - 6 * x + 8
    y = y/y.max()
    tip = 0.2
    y = y - tip + tip * np.random.rand(300)
    x = x.reshape(300,1)
    y = y.reshape(300,1)
    img_savepath = "C:\\Users\\asus\\Desktop\\projects\\tmpimgs"
    gif_name = "C:\\Users\\asus\\Desktop\\projects\\DNNNet.gif"

    hid_dim_list = [1,10,30,1]
    num_epoches = 5000
    stepsize = 100
    lamda = -0.1
    stepsize = 100
    base_lr = 0.001
    max_lr = 0.005
    learning_rate = get_triangular_lr(0, stepsize, base_lr, max_lr)
    dnnnet = DNNNet(hid_dim_list)
    if not os.path.exists(img_savepath):
    	os.mkdir(img_savepath)
    plt.ion()
    for iteration in range(num_epoches):
        weight_list,bais_list = dnnnet.backward(x,y)
        dnnnet.update_grad_sgd(weight_list,bais_list,learning_rate,lamda)
        loss = dnnnet.loss(x,y)
        if iteration%stepsize == 0:
        	# SGDR algorithm
        	learning_rate = get_triangular_lr(iteration,stepsize,base_lr,max_lr)
        	plt.cla()
        	y_pre = dnnnet.forward(x)
        	plt.scatter(x,y,c = "r",marker='o', edgecolors='g')
        	sns.despine()
        	plt.plot(x,y_pre)
        	plt.pause(0.1)
        	temp_file = os.path.join(img_savepath,"pic" + str(iteration//100) + ".png")
        	plt.savefig(temp_file)
        	print("loss: ",loss)
    plt.ioff()
    utils.save_gif(img_savepath,gif_name)
    if os.path.exists(img_savepath):
    	shutil.rmtree(img_savepath,True)
if __name__ == '__main__':
	main()