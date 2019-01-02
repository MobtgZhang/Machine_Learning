import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
 
import imageio
import os
import shutil

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
class BPNetWork(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super(BPNetWork,self).__init__()
        self.hidden = nn.Linear(in_size,hidden_size)
        self.output = nn.Linear(hidden_size,out_size)
    def forward(self,x):
        hid = torch.sigmoid(self.hidden(x))
        return self.output(hid)
def train():
    x_data = torch.linspace(-1,1,300).unsqueeze(1)
    y_data = 10 * torch.pow(x_data,3) - 6 * x_data + 8
    y_data = y_data/y_data.max()
    tip = 0.03
    y_data = y_data - tip + tip * torch.randn(x_data.size())
    x_data,y_data = Variable(x_data),Variable(y_data)
    in_size = 1
    hidden_size = 50
    out_size = 1
    bpnet = BPNetWork(in_size,hidden_size,out_size)
    optimizer = optim.Adam(bpnet.parameters(),lr = 0.4)
    loss_fun = nn.MSELoss()
    file_path = "pictures"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    plt.ion()
    for epoch in range(800):
        optimizer.zero_grad()
        predict = bpnet(x_data)
        loss = loss_fun(predict,y_data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            plt.cla()
            plt.scatter(x_data.data.numpy(),y_data.data.numpy())
            plt.plot(x_data.data.numpy(),predict.data.numpy(),"-r",lw = 3)
            print(epoch,loss.data.item())
            plt.pause(0.1)
            filename = os.path.join(file_path,"pic" + str(epoch//50) + ".png")
            plt.savefig(filename)
    plt.ioff()
    save_gif(file_path,"pytorch_bpnet.gif")
    shutil.rmtree(file_path,True)
    print ("Directory: " + file_path +" was removed!")
if __name__ == "__main__":
    train()
