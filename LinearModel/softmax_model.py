import numpy as np
def softmax(x):
	out = np.exp(x)
	for k in range(len(out)):
		out[k] = out[k]/out[k].sum()
	return out
def changeNum(data,list_num):
	matrix = np.zeros(data.shape)
	for k in range(len(list_num)):
		matrix[k,list_num[k]] = 1
	return matrix
class SoftMax:
	def __init__(self,in_size,out_size):
		self.in_size = in_size
		self.out_size = out_size
		self.nums = 0
		self.weight = np.random.randn(self.in_size,self.out_size)
		self.bais = np.random.randn(self.out_size)
	def forward(self,x):
		out = softmax(np.dot(x,self.weight) + self.bais)
		return out
	def backward(self,x,y_true):
		self.nums = len(x)
		y_pred = self.forward(x)
		weight_grad = -np.matmul(x.T,(y_true - y_pred))
		bais_grad = y_true - y_pred
		return weight_grad,bais_grad
	def loss(self,x,y_true):
		y_pred = self.forward(x)
		return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
	def update_grad(self,weight_grad,bais_grad,learning_rate,lamda):
		N = self.nums
		self.weight = self.weight - learning_rate*(weight_grad/N + lamda*self.weight)
		self.bais = self.bais - learning_rate*bais_grad/N
def main():
	in_size = 10
	length = 5
	out_size = 15
	num_epoches = 5
	learning_rate = 0.8
	lamda = -0.1
	x = np.random.randn(length,in_size)
	y = np.zeros((length,out_size))
	indexes = [np.random.randint(0,out_size) for k in range(length)]
	for k in range(length):
		y[k,indexes[k]] = 1
	softmax = SoftMax(in_size,out_size)
	for k in range(num_epoches):
		weight_grad,bais_grad = softmax.backward(x,y)
		softmax.update_grad(weight_grad,bais_grad,learning_rate,lamda)
		loss = softmax.loss(x,y)
		print("loss: ",loss)
	z = softmax.forward(x)
	list_num = np.argmax(z,axis = 1)
	data = changeNum(z,list_num)
	print(data)
if __name__ == '__main__':
	main()