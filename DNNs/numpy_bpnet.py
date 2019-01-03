import numpy as np
import utils
class Linear:
	def __init__(self,in_dim,out_dim):
		self.in_dim = in_dim
		self.out_dim = out_dim
		# initalize the weight and bais in_dim*out_dim
		self.weight =  np.random.rand(in_dim,out_dim)
		self.bais = np.random.rand(out_dim)
	def forward(self,x):
		# input the value length*in_dim
		y = np.matmul(x,self.weight)+self.bais
		return y
	def __str__(self):
		return "Linear layer: \nin_dim: "+str(self.in_dim)+"," + "out_dim: "+str(self.out_dim)
class DNNNet:
	def __init__(self,hid_dim_list):
		self.hid_dim_list = hid_dim_list
		self.hidden_list = []
		self.num_layers = len(hid_dim_list)-1
		for k in range(self.num_layers):
			fc = Linear(self.hid_dim_list[k],self.hid_dim_list[k+1])
			self.hidden_list.append(fc)
	def forward(self,x):
		for k in range(self.num_layers):
			activate = utils.sigmoid(x)
			x = self.hidden_list[k].forward(activate)
		activate = utils.sigmoid(x)
		return activate
	def backward(self,x,y_true):
		bais_list = []
		weight_list = []
		for k in range(self.num_layers,0,-1):
			# calculdate delta
			if k==self.num_layers:
				out = x
				for j in range(k):
					activate = utils.sigmoid(out)
					out = self.hidden_list[j].forward(activate)
				delta = utils.diff_sigmoid(out)*(utils.sigmoid(out)-y_true) # 9*7
			else:
				# update the delta
				weight_post = self.hidden_list[k].weight
				delta_post = bais_list[self.num_layers - k-1]
				out = x
				for j in range(k):
					activate = utils.sigmoid(out)
					out = self.hidden_list[j].forward(activate)
				delta = utils.diff_sigmoid(out)*np.matmul(delta_post,weight_post.T)
			# print(delta.shape)
			# calculdate weight
			out = x
			for j in range(k-1):
				activate = utils.sigmoid(out)
				out = self.hidden_list[j].forward(activate)
			activate = utils.sigmoid(out)
			weight_grad = np.matmul(activate.T,delta) # 10*7

			# add bais_list and weight_list
			bais_list.append(delta)
			weight_list.append(weight_grad)
		return weight_list,bais_list
	def update_grad(self,weight_list,bais_list):
		for k in range(self.num_layers):
			# print(self.hidden_list[k].weight.shape,weight_list[k].shape)
			# print(self.hidden_list[k].bais.shape,bais_list[k].shape)
			self.hidden_list[k].weight = self.hidden_list[k].weight - weight_list[self.num_layers - k-1]
			self.hidden_list[k].bais = self.hidden_list[k].bais - bais_list[self.num_layers - k-1]
	def loss(self,x,y_true):
		y_pred = self.forward(x)
		return utils.quadlf(y_pred,y_true)
	def __str__(self):
		line = ""
		for k in range(self.num_layers):
			line = line + "("+str(k)+")" +"Linear layer: \nin_dim: "+str(self.hidden_list[k].in_dim)+"," + "out_dim: "+str(self.hidden_list[k].out_dim) +"\n"
		return line
def main():
	x = np.random.rand(9,5)
	y = np.random.rand(9,7)
	linear = Linear(5,10)
	hid_dim_list = [5,10,20,60,100,50,30,10,7]
	num_epoches = 20
	dnnnet = DNNNet(hid_dim_list)
	for k in range(num_epoches):
		weight_list,bais_list = dnnnet.backward(x,y)
		dnnnet.update_grad(weight_list,bais_list)
		loss = dnnnet.loss(x,y)
		print("loss: ",loss)
if __name__ == '__main__':
	main()
