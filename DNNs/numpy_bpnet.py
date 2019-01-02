import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))
def diff_sigmoid(x):
	return sigmoid(x)(1-sigmoid(x))
class Linear:
	def __init__(self,in_dim,out_dim):
		# initalize the weight and bais in_dim*out_dim
		self.weight =  np.random.rand(in_dim,out_dim)
		self.bais = np.random.rand(out_dim)
	def forward(self,x):
		# input the value length*in_dim
		y = np.matmul(x,self.weight)+self.bais
		return y
class NetDNN:
	def __init__(self,in_dim,hid_dim_list,out_dim):
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hid_dim_list = hid_dim_list
		self.num_layers = len(hid_dim_list) - 1
		self.hidden_layers = []
		# input layer
		self.input  = Linear(self.in_dim,self.hid_dim_list[0])
		# hidden layers
		for k in range(self.num_layers):
			fc = Linear(self.hid_dim_list[k],self.hid_dim_list[k+1])
			self.hidden_layers.append(fc)
		# output layers
		self.output = Linear(self.hid_dim_list[self.num_layers],out_dim)
	def forward(self,x):
		hid = sigmoid(self.input.forward(x))
		for k in range(self.num_layers):
			hid = self.hidden_layers[k].forward(hid)
			hid = sigmoid(hid)
		out = self.output.forward(hid)
		return out
	def _layer_forward(x,index):
		hid = sigmoid(self.input.forward(x))
		for k in range(index):
			hid = self.hidden_layers[k].forward(hid)
			hid = sigmoid(hid)
		if index <0:
			return self.input.forward(x)
		else:
			return self.hidden_layers[index]
	def loss(self,x,y_score,y_label):
		weight_list = []
		bais_list = []
		# the last layer gradient
		out_grad = (y_score - y_label)
		weight_list.append(out_grad)

		delta = (y_score - y_label)
		bais_list.append(delta)

		weight = self.output.weight
		# the hidden layers gradient
		for k in range(self.num_layers-1,0,-1):
			# delta value
			m = diff_sigmoid(self._layer_forward(x,k))
			n = weight.T.matmul(delta)
			delta = np.multiply(m,n)
			
			m = sigmoid(self._layer_forward(x,k-1))
			out_grad = delta.matmul(m.T)

			bais_list.append(delta)
			weight_list.append(out_grad)
			# weight value
			weight = self.hidden_layers[k].weight
		# the input layers gradient
		
	def update_grad(self,weight_list,bais_list,learning_rate):
		# input layer
		self.input.weight = self.input.weight - learning_rate*weight_list[0]
		self.input.bais = self.input.bais - learning_rate*bais_list[0]
		# hidden layers
		for k in range(self.num_layers):
			self.hidden_layers[k].weight - learning_rate*weight_list[k]
			self.hidden_layers[k].bais - learning_rate*bais_list[k]
		# output layer
		self.output.weight = self.input.weight - learning_rate*weight_list[self.num_layers]
		self.output.weight = self.input.bais - learning_rate*bais_list[self.num_layers]
	def backward(self):
		out = self
def main():
	
	in_dim = 5
	out_dim = 4
	hid_dim_list = [5,5,3,4,7,8,6,8]
	bpnet = NetDNN(in_dim,hid_dim_list,out_dim)
	x = np.random.rand(9,5)
	y = bpnet.forward(x)
	print(y)
	print(y.shape)
	'''
	matrix = np.random.rand(9,5)
	linear = np.random.rand(1,5)
	print(linear)
	print("\n")
	print(matrix)
	print("\n")
	out = matrix + linear
	print(out)
	print("\n")
	'''
if __name__ == "__main__":
	main()
