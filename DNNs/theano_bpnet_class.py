import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

class linear:
	def __init__(self,in_size,out_size,activity_function = None):
		self.input_size = in_size
		self.output_size = out_size
		self.weight = theano.shared(np.random.normal(0,1,(in_size,out_size)))
		self.bais = theano.shared(np.zeros((out_size,)) + 0.1)
		self.act_func = activity_function
		# gradient 
		self.gW = None
		self.gb = None
	def forward(self,x):
		W_plus_b = T.dot(x,self.weight) + self.bais
		if self.act_func is None:
			return W_plus_b
		else:
			return self.act_func(W_plus_b)
	def update_grad(self,cost):
		self.gW,self.gb = T.grad(cost,[self.weight,self.bais])
		return (self.weight,self.gW),(self.bais,self.gb)
class bpnet:
	def __init__(self,input_size,hidden_size,output_size,activity_function = None):
		if activity_function is not None:
			self.act_func = activity_function
		else:
			self.act_func = T.tanh
		self.hidden = linear(input_size,hidden_size,self.act_func)
		self.predict = linear(hidden_size,output_size,None)
	def forward(self,x):
		temp = self.hidden.forward(x)
		return self.predict.forward(temp)
	def update_grad(self,cost,learning_rate):
		temp = []
		tupleH = []
		listA,listB = self.hidden.update_grad(cost)
		tupleH.append(listA)
		tupleH.append(listB)
		listA,listB = self.predict.update_grad(cost)
		tupleH.append(listA)
		tupleH.append(listB)
		for k in range(len(tupleH)):
			temp.append((tupleH[k][0],tupleH[k][0]-learning_rate * tupleH[k][1]))
		return temp
def Main():

	# make up some data 
	x_data = np.linspace(-1,1,300)[:,np.newaxis]
	noise = np.random.normal(0,0.05,x_data.shape)
	y_data = np.square(x_data) - 0.5 + noise
	
	# determine inputs dtype 
	x = T.dmatrix("x")
	y = T.dmatrix("y")

	# defination of the layers 
	input_size = 1
	hidden_size = 10
	output_size = 1
	learning_rate = 0.1
	net = bpnet(input_size,hidden_size,output_size)

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
