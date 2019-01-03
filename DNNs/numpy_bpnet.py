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
		return x
	def backward(self,x,y_true):
		bais_list = []
		weight_list = []
		y_pred = self.forward(x)
		delta = utils.diff_quadlf(y_true,y_pred)
		bais_list.append(delta)
		for k in range(self.num_layers-2,0,-1):
			# calculate this layer grad delta and weight_grad
			out = x
			for j in range(k):
				activate = utils.sigmoid(out)
				out = self.hidden_list[j].forward(activate)
			
			activate = utils.sigmoid(out)
			weight_grad = np.matmul(activate.T,delta)
			
			weight_list.append(weight_grad)
			# calculdate the previews layer delta
			t = np.matmul(delta,self.hidden_list[k+1].weight.T)
			if k==0:
				pass
			else:
				out = x
				for j in range(k-1):
					activate = utils.sigmoid(out)
					out = self.hidden_list[j].forward(activate)
			df_a = utils.diff_sigmoid(out)
			delta = np.multiply(df_a,t)
			bais_list.append(delta)
		print(len(bais_list))
	def update_grad(self,bais_list,weight_list):
		pass
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
	dnnnet = DNNNet(hid_dim_list)
	z = dnnnet.forward(x)
	dnnnet.backward(x,y)
	print(z)
if __name__ == '__main__':
	main()
