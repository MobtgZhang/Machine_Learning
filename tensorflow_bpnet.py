import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def addLayer(inputData,in_size,out_size,activity_function = None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size])+0.1)
	Weight_plus_b = tf.matmul(inputData,Weights) + biases
	if activity_function ==None:
		output = Weight_plus_b
	else:
		output = activity_function(Weight_plus_b)
	return output
def Main():
	x_data = np.linspace(-1,1,300)[:,np.newaxis]
	y_data = 10 * np.power(x_data,3) - 5 * x_data + 6
	y_data = y_data / y_data.max()
	tip = 0.1
	y_data = y_data - tip + tip * np.random.random(y_data.shape)

	in_data = tf.placeholder(tf.float32,shape=(None,1))
	sign_data = tf.placeholder(tf.float32,shape=(None,1))

	layer = addLayer(in_data,1,10,activity_function = tf.nn.tanh)
	prediction = addLayer(layer,10,1,None)

	loss = tf.reduce_mean(tf.reduce_sum(tf.square(sign_data - prediction),reduction_indices = [1]))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	init = tf.initialize_all_variables()
	sess = tf.Session()

	sess.run(init)
	plt.ion()
	for k in range(15000):
		sess.run(train_step,feed_dict = {in_data:x_data,sign_data:y_data})
		loss_data = sess.run(loss,feed_dict = {in_data:x_data,sign_data:y_data})
		if k %50 ==0:
			plt.cla()
			pre_t = sess.run(prediction,feed_dict = {in_data:x_data})
			x_t = x_data.reshape(x_data.shape[0])
			y_t = pre_t.reshape(pre_t.shape[0])
			z_t = y_data.reshape(y_data.shape[0])
			plt.scatter(x_t,z_t)
			plt.plot(x_t,y_t,'r-',lw = 3)
			plt.pause(0.01)
		print(loss_data)
	plt.ioff()
if __name__ =="__main__":
	Main()
