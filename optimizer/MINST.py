import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
import torch.utils.data
import os
import time
def select_optim(name,model,learning_rate):
    '''
    SGD,Momentum,Nesterov,RMSprop,AdaGrad,AdaGrad,AdaDelta,Adam,AdamAMSGrad,AdamW,AdamWAMSGrad
    :param name:
    :param model:
    :param learning_rate:
    :return:
    '''
    if name == "SGD":
        return optim.SGD(model.parameters(),lr=learning_rate)
    elif name == "Momentum":
        return optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
    elif name == "Nesterov":
        return optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,nesterov=True)
    elif name == "RMSprop":
        return optim.RMSprop(model.parameters(),lr = learning_rate)
    elif name == "AdaGrad":
        return optim.Adagrad(model.parameters(),lr= learning_rate)
    elif name == "AdaDelta":
        return optim.Adadelta(model.parameters(),lr= learning_rate)
    elif name == "Adam":
        return optim.Adam(model.parameters(),lr=learning_rate)
    elif name == "AdamAMSGrad":
        return optim.Adam(model.parameters(), lr=learning_rate,amsgrad=True)
    elif name == "Adamax":
        return optim.Adamx(model.parameters(), lr=learning_rate,)
    elif name == "AdamW":
        return optim.AdamW(model.parameters(),lr=learning_rate)
    elif name == "AdamWAMSGrad":
        return optim.AdamW(model.parameters(), lr=learning_rate,amsgrad=True)
    else:
        raise TypeError("Unknown model type:%s"%str(name))
def train(model,train_loader,optimizer,epoches,gpu):
    loss_sum = 0
    for batch_idx,(data,target) in enumerate(train_loader):
        if gpu:
            data,target = Variable(data).cuda(),Variable(target).cuda()
        else:
            data, target = Variable(data),Variable(target)
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss_sum += loss.cpu().data.numpy()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoches, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.cpu().data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_sum /= len(train_loader)
    return loss_sum
def test(model,test_loader,gpu):
    test_loss = 0
    correct = 0
    for data,target in test_loader:
        if gpu:
            data, target = Variable(data).cuda(), Variable(target).cuda()
        else:
            data, target = Variable(data), Variable(target)
        output = model(data)
        pred = torch.argmax(output.cpu().data,dim=1)
        loss = F.cross_entropy(output,target)
        test_loss += loss.cpu().data
        correct += pred.eq(target.cpu().data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
# make the network
class L5Net(nn.Module):
    def __init__(self):
        super(L5Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,5)
        self.conv3 = nn.Conv2d(20,40,3)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(40,10)
    def forward(self,input):
        batch_size = input.size(0)
        hid = F.relu(self.mp(self.conv1(input)))
        hid = F.relu(self.mp(self.conv2(hid)))
        hid = F.relu(self.mp(self.conv3(hid)))

        predict = hid.view(batch_size,-1)

        output = self.fc(predict)

        return F.softmax(output)
def main():
    # preparing parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data/", help="Defination of the data file path.", type=str)
    parser.add_argument("--gpu",default=True,help="Whether using the cuda.",type=bool)
    parser.add_argument("--learning_rate", default=0.01, help="Defination of the network training learning rate.",
                        type=float)
    parser.add_argument("--epoches", default=10, help="Defination of loop numbers.", type=int)
    parser.add_argument("--batch_size", default=64, help="Batchfiy the data.", type=int)
    parser.add_argument("--compare",default=False,help="Compare of different optimizers.",type=bool)
    args = parser.parse_args()
    # preparing data set
    # MNIST Dataset
    train_dataset = datasets.MNIST(root=args.datadir,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root=args.datadir,
                                  train=False,
                                  transform=transforms.ToTensor())
    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, shuffle=False)

    print(args)
    if args.compare:
        if not os.path.exists("log"):
            os.mkdir("log")
        optim_list = ["SGD","Momentum","Nesterov","RMSprop","AdaGrad","AdaGrad","AdaDelta","Adam","AdamAMSGrad","AdamW","AdamWAMSGrad"]
        time0 = time.time()
        time_file = open(os.path.join("log","time_costs"),mode="w")
        for item in optim_list:
            model = L5Net()
            if args.gpu:
                model.cuda()
            optimizer = select_optim(item,model,args.learning_rate)
            with open(os.path.join("log",item),mode="w",encoding="utf-8") as f:
                for iter in range(args.epoches):
                    loss = train(model,train_loader,optimizer,iter,args.gpu)
                    f.write(str(loss) + "\n")
            time1 = time.time()
            time_file.write(item + " : " + str(time1-time0)+"\n")
        time_file.close()
        # draw

    else:
        model = L5Net()
        if args.gpu:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,momentum=0.8)
        for iter in range(args.epoches):
            train(model,train_loader,optimizer,iter,args.gpu)
            test(model,test_loader,args.gpu)
if __name__ == '__main__':
    main()








