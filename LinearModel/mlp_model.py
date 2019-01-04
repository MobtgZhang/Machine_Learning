import numpy as np
import matplotlib.pyplot as plt
class MlpModel:
	def __init__(self,in_dim):
		self.in_dim = in_dim
		# initalize the weight and bais in_dim*out_dim
		self.weight =  np.random.rand(in_dim,1)
		self.bais = np.random.rand(1)
	def forward(self,x):
		# input the value length*in_dim
		y = np.matmul(x,self.weight)+self.bais
		return y
	def backward(self,x,y_true):
		y_pred = self.forward(x)
		if np.dot(y_true,y_pred) > 0:
			weight_grad = 0
			bais_grad = 0
		else:
			weight_grad = -np.dot(y_true,x)
			bais_grad = -y_true
		return weight_grad,bais_grad
	def update_grad(self,weight_grad,bais_grad,learning_rate,lamda):
		self.weight = self.weight - (learning_rate*weight_grad + lamda*self.weight)
		self.bais = self.bais - learning_rate*bais_grad
	def loss(self,x,y_true):
		y_pred = self.forward(x)
		return np.mean(np.power(y_pred - y_true,2))
	def __str__(self):
		return "Linear layer: \nin_dim: "+str(self.in_dim)+"," + "out_dim: "+str(self.out_dim)
def main():
	in_size = 8
	length = 6
	x = np.random.randn(length,in_size)
	y = np.around(np.random.random((length)))
	num_epoches = 50
	learning_rate = 0.8
	lamda = 0.1
	model = MlpModel(in_size)
	out = model.forward(x)
	for k in range(num_epoches):
		weight_list,bais_list = model.backward(x,y)
		model.update_grad(weight_list,bais_list,learning_rate,lamda)
		loss = model.loss(x,y)
		print("loss: ",loss)
if __name__ == '__main__':
	main()