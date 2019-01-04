import numpy as np
import imageio
import os
def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)
def save_gif(file_path,gif_name):
    image_list = []
    for f in os.listdir(file_path):
        image_list.append(os.path.join(file_path,f))
    create_gif(image_list, gif_name)
# sigmoid functions
def sigmoid(x):
	return 1/(1+np.exp(-x))
def diff_sigmoid(x):
	return np.multiply(sigmoid(x),(1-sigmoid(x)))
# tanh functions
def tanh(x):
	return 2*sigmoid(2*x)-1
def diff_tanh(x):
	return 4*sigmoid(2*x)
# relu functions
def relu(x):
	x[x<0] = 0
	return x
def diff_relu(x):
	x[x<0] = 0
	x[x>0] = 1
	return x
# softplus functions
def softplus(x):
	return np.log(1+np.exp(x))
def diff_softplus(x):
	return sigmoid(x)
# quadratic loss function
def quadlf(y_true,y_pred):
	return np.sum(np.power(y_pred - y_true,2))
def diff_quadlf(y_true,y_pred):
	return y_pred - y_true
# cross-entropy loss function
def crossentropylf(y_true,y_pred):
	return -np.sum(y_true*np.log(y_pred))
def main():
	x = np.random.rand(5) - 0.5
	print(diff_relu(x))
if __name__ == '__main__':
	main()