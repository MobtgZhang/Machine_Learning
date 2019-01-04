import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
import shutil
class Linear:
	def __init__(self,in_size,out_size):
		self.in_size = in_size
		self.out_size = out_size
		self.weight = tf.Variable(tf.random_normal([in_size,out_size]))
		self.bais = tf.Variable(tf.zeros([1,out_size])+0.1)
	def forward(self,x):
		Weight_plus_b = tf.matmul(x,self.weight) + self.bais
		return Weight_plus_b
class DNNNet:
	def __init__(self,hid_dim_list,out_size):
		self.hid_dim_list = hid_dim_list
		self.out_size = out_size
		self.num_layers = len(self.hid_dim_list) - 1
		self.hidden_list = []
		for k in range(self.num_layers):
			fc = Linear(self.hid_dim_list[k],self.hid_dim_list[k+1])
			self.hidden_list.append(fc)
		self.output = Linear(self.hid_dim_list[self.num_layers],self.out_size)
	def forward(self,x):
		out = x
		for k in range(self.num_layers):
			out = self.hidden_list[k].forward(out)
			out = tf.nn.tanh(out)
		out = self.output.forward(out)
		return out
def main():
	x_data = np.linspace(-1,1,300)[:,np.newaxis]
	y_data = 10 * np.power(x_data,3) - 5 * x_data + 6
	y_data = y_data / y_data.max()
	tip = 0.1
	y_data = y_data - tip + tip * np.random.random(y_data.shape)

	in_data = tf.placeholder(tf.float32,shape=(None,1))
	sign_data = tf.placeholder(tf.float32,shape=(None,1))
	hid_dim_list = [1,10,20,30]
	out_size = 1
	learning_rate = 0.01
	num_epoches = 2000
	img_savepath = "C:\\Users\\asus\\Desktop\\projects\\tmpimgs"
	gif_name = "C:\\Users\\asus\\Desktop\\projects\\tf_DNNNet.gif"
	dnnnet = DNNNet(hid_dim_list,out_size)

	prediction = dnnnet.forward(in_data)
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(sign_data - prediction),reduction_indices = [1]))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	plt.ion()
	if not os.path.exists(img_savepath):
		os.mkdir(img_savepath)
	for k in range(num_epoches):
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
			temp_file = os.path.join(img_savepath,"pic" + str(k//100) + ".png")
			plt.savefig(temp_file)
			print(loss_data)
	plt.ioff()
	utils.save_gif(img_savepath,gif_name)
	if os.path.exists(img_savepath):
		shutil.rmtree(img_savepath,True)
if __name__ =="__main__":
	main()
