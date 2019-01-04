import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

class Linear:
    def __init__(self,in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = theano.shared(np.random.normal(0,1,(in_dim,out_dim)))
        self.bais = theano.shared(np.zeros((out_dim,))+0.1)
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
	def __init__(self,hid_dim_list,activate_fun = None):
		self.hid_dim_list = hid_dim_list
		self.num_layers = len(hid_dim_list) - 1
		self.hidden_list = []
		if activate_fun is not None:
			self.act_func = activate_fun
		else:
			raise("unknown activate function!")
		for k in range(self.num_layers):
			fc = Linear(self.hid_dim_list[k],self.hid_dim_list[k+1])
			self.hidden_list.append(fc)
	def forward(self,x):
		out = x
		for k in range(self.num_layers):
			out = self.hidden_list[k].forward(out)
			if type(self.act_func) is list and len(self.act_func) == self.num_layers:
				out = self.act_func[k](out)
			elif type(self.act_func) is theano.tensor.elemwise.Elemwise:
				out = self.act_func(out)
			else:
				raise("unknown activate function!")
		return out
	def backward(self,x,y,cost,grad_list):
		train = theano.function(
			inputs = [x,y],
			outputs = [cost],
			updates = grad_list)
		return train(x,y)
	def update_grad(self,cost,learning_rate):
		grad_list = []
		for k in range(self.num_layers):
			weight,gW,bais,gb = self.hidden_list[k].update_grad(cost)
			grad_list.append((weight,weight - learning_rate*gW))
			grad_list.append((bais,bais - learning_rate*gb))
		return grad_list
def main():
	# make up some data 
	x_data = np.linspace(-1,1,300)[:,np.newaxis]
	noise = np.random.normal(0,0.05,x_data.shape)
	y_data = np.square(x_data) - 0.5 + noise
	
	# determine inputs dtype 
	x = T.dmatrix("x")
	y = T.dmatrix("y")
	# defination of the layers
	hid_dim_list = [1,20,30,10,1]
	learning_rate = 0.1
	num_epoches = 500
	net = DNNthE(hid_dim_list,T.tanh)
	# define the cost
	prediction = net.forward(x)
	cost = T.mean(T.square(prediction - y))
	# update the grad 
	grad_list = net.update_grad(cost,learning_rate)
	# apply gradient descent
	train = theano.function(
		inputs = [x,y],
		outputs = [cost],
		updates = grad_list)
	# prediction 
	predict = theano.function(inputs = [x],outputs = prediction)
	# y_pred = predict(x).reshape(300)
	exit()
	# traininig model
	plt.ion()
	for k in range(num_epoches):
		err = net.backward(x,y,cost,grad_list)
		if k%50 == 0:
			y_pred = predict(x).reshape(300)
			plt.cla()
			plt.scatter(x,y)
			plt.plot(x,y_pred,c='g',lw = 4)
			plt.pause(0.1)
			print(err[0])
	plt.ioff()
if __name__ == "__main__":
	main()