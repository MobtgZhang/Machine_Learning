import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
	return 1/(1+np.exp(-x))
def diff_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))
class Logistic:
	def __init__(self,in_size):
		self.in_size = in_size
		self.nums = 0
		self.weight = np.random.randn(in_size,1)
		self.bais = np.random.randn(1)
	def forward(self,x):
		return sigmoid(np.dot(x,self.weight) + self.bais)
	def backward(self,x,y_true):
		self.nums = len(x)
		y_pred = self.forward(x)
		weight_grad = -np.matmul(x.T,(y_true - y_pred))
		bais_grad = y_true - y_pred
		return weight_grad,bais_grad
	def loss(self,x,y_true):
		y_pred = self.forward(x)
		return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
	def update_grad(self,weight_grad,bais_grad,learning_rate,lamda = 0):
		self.weight = self.weight - learning_rate*(weight_grad + lamda*self.weight)
		self.bais = self.bais - learning_rate*bais_grad
def main():
	in_size = 8
	length = 6
	x = np.random.randn(length,in_size)
	y = np.around(np.random.random((length,1)))
	num_epoches = 3
	learning_rate = 0.8
	lamda = 0.1
	logistic = Logistic(in_size)
	for k in range(num_epoches):
		weight_list,bais_list = logistic.backward(x,y)
		logistic.update_grad(weight_list,bais_list,learning_rate,lamda)
		loss = logistic.loss(x,y)
		print("loss: ",loss)
if __name__ == '__main__':
	main()