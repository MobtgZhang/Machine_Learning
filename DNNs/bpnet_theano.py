import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

class linear:
	def __init__(self,x,in_size,out_size,activition_function = None):
		self.W = theano.shared(np.random.normal(0,1,(in_size,out_size)))
		self.b = theano.shared(np.zeros((out_size,)) + 0.1)
		self.act_func = activition_function
		self.output = T.dot(x,self.W) + self.b
		if self.act_func is None:
			self.output = T.dot(x,self.W) + self.b
		else:
			self.output = self.act_func(self.output)

# make up some data 
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise



# determine the inputs dtype
x = T.dmatrix("x")
y = T.dmatrix("y")

# add layers
in_size = 1
hidden_size = 10
out_size = 1
l1 = linear(x,in_size,hidden_size,T.tanh)
l2 = linear(l1.output,hidden_size,out_size,None)

# compute the cost
cost = T.mean(T.square(l2.output - y))

gW1,gb1,gW2,gb2 = T.grad(cost,[l1.W,l1.b,l2.W,l2.b])

# apply gradient descent
learning_rate = 0.1
train = theano.function(
	inputs = [x,y],
	outputs = [cost],
	updates = [(l1.W,l1.W - learning_rate * gW1),
				(l1.b,l1.b - learning_rate * gb1),
				(l2.W,l2.W - learning_rate * gW2),
				(l2.b,l2.b - learning_rate * gb2)]
	)
# prediction
predict = theano.function(inputs = [x],outputs = l2.output)

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
