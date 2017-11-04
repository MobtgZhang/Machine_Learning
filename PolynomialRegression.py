from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import copy
# using backpropagation nerual network for three layers
class linear:
	def __init__(self,in_size,out_size,activity_function):
		self.input_size = in_size
		self.output_size = out_size
		self.weight = tf.Variable(tf.random_normal([self.input_size,self.output_size]))
		self.bais = tf.Variable(tf.zeros([1,self.output_size])+0.1)
		self.act_func = None
		label = activity_function.strip()
		if label == "tanh":
			self.act_func = tf.nn.tanh
		elif label == "sigmoid":
			self.act_func = tf.nn.sigmoid
		elif label == "relu":
			self.act_func = tf.nn.relu
		elif label == "softplus":
			self.act_func = softplus
		elif label == "none":
			pass
		else:
			raise Exception("Unknown activition function type")
	def forward(self,in_data):
		Weight_plus_b = tf.matmul(in_data,self.weight) + self.bais
		if self.act_func == None:
			output = Weight_plus_b
		else:
			output = self.act_func(Weight_plus_b)
		return output
class bpnet:
	def __init__(self,input_size,hiddden_size,output_size,activity_function = "tanh"):
		self.hiddden = linear(input_size,hiddden_size,activity_function)
		self.predict = linear(hiddden_size,output_size,"none")
	def forward(self,x):
		layer = self.hiddden.forward(x)
		output = self.predict.forward(layer)
		return output
def train_model(model,x_data,y_data):
	epoches = 500
	learning_rate = 0.01
	# define a bpnet
	in_data = tf.placeholder(tf.float32,shape = (None,input_size))
	sign_data = tf.placeholder(tf.float32,shape = (None,output_size))
	prediction = model.forward(in_data)
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(sign_data - prediction),
			reduction_indices = [1]))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	plt.ion()
	for k in range(epoches):
		sess.run(train_step,feed_dict = {in_data:x_data,sign_data:y_data})
		loss_data = sess.run(loss,feed_dict = {in_data:x_data,sign_data:y_data})
		if k%50 ==0:
			plt.cla()
			pre_t = sess.run(prediction,feed_dict = {in_data:x_data})
			x_t = x_data.reshape(x_data.shape[0])
			y_t = pre_t.reshape(pre_t.shape[0])
			z_t = y_data.reshape(y_data.shape[0])
			plt.scatter(x_t,z_t)
			plt.plot(x_t,y_t,"r-",lw = 3)
			plt.pause(0.01)
		print(loss_data)
	plt.ioff()
	sess.close()
def Main():
	
	x_data = np.linspace(-1,1,300)[:,np.newaxis]
	y_data = 10 * np.power(x_data,3) - 5 * x_data + 6
	y_data = y_data / y_data.max()
	tip = 0.1
	y_data = y_data - tip + tip * np.random.random(y_data.shape)
	# uisng polynomial features regression
	model = make_pipeline(PolynomialFeatures(degree = 3,interaction_only = False,include_bias = True),
			Ridge())
	model.fit(x_data,y_data) 
	# using nerual network training data
	input_size = 1
	hiddden_size = 10
	output_size = 1
	net = bpnet(input_size,hiddden_size,output_size)
	plt.figure()


	epoches = 5000
	learning_rate = 0.01
	# define a bpnet
	in_data = tf.placeholder(tf.float32,shape = (None,input_size))
	sign_data = tf.placeholder(tf.float32,shape = (None,output_size))
	prediction = net.forward(in_data)
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(sign_data - prediction),
			reduction_indices = [1]))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	plt.ion()
	for k in range(epoches):
		sess.run(train_step,feed_dict = {in_data:x_data,sign_data:y_data})
		loss_data = sess.run(loss,feed_dict = {in_data:x_data,sign_data:y_data})
		if k%50 ==0:
			plt.cla()
			pre_t = sess.run(prediction,feed_dict = {in_data:x_data})
			x_t = x_data.reshape(x_data.shape[0])
			y_t = pre_t.reshape(pre_t.shape[0])
			z_t = y_data.reshape(y_data.shape[0])
			plt.scatter(x_t,z_t)
			plt.plot(x_t,y_t,"r-",lw = 3)
			plt.pause(0.01)
		print(loss_data)
	plt.ioff()
	# The resluts:
	y = model.predict(x_data)
	x = copy.deepcopy(x_data)
	pre_t = sess.run(prediction,feed_dict = {in_data:x_data})
	z = pre_t.reshape(pre_t.shape[0])
	fig,ax = plt.subplots()  
	plt.scatter(x_data.reshape(300),y_data.reshape(300),c = "g",label = "the source data")
	plt.plot(x,y,c = "r",label = "polynominal analysis",lw = 3)
	plt.plot(x,z,c = "b",label = "nerual network analysis",lw = 3)
	plt.legend(loc = "best")
	plt.show()
	sess.close()
if __name__ == "__main__":
	Main()