import argparse
import logging
import os
import time
from data import DataExtractor,download
from model import DNNNet


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def train(model,loadedData,args):
    logger = logging.getLogger()
    logger.info("Training for %s wine data."%args.type)
    for k in range(args.epoches):
        loss_sum = 0
        length = loadedData.getlen(0)
        for j in range(length):
            input,target = loadedData.getitem(j, 0)
            loss = model.train(input,target,args.learning_rate,args.lambd)
            loss_sum = loss_sum + loss
        loss = loss_sum/length
        logger.info("Training loss value:%f"%(loss))
        # validate the result
        input,target = loadedData.getvalidate()
        output = model.forward(input)
        loss = model.loss(output,target)
        logger.info("Validating loss value:%f" % (loss))
    logger.info("Trained for %s wine data completed." % args.type)
    modeldir = os.path.join(args.logdir,"%s.pt"%model.name)
    model.save_dict_parameters(modeldir)
    logger.info("Model saved in path:%s"%modeldir)
def test(model,DataSet):
    logger = logging.getLogger()
    input,target = DataSet.gettest()
    output = model.forward(input)
    loss = model.loss(output,target)
    logger.info("Testing loss value:%f" % (loss))
def main():
    # parse cmdline argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data", help="Defination of the data file path.", type=str)
    parser.add_argument("--learning_rate", default=0.3, help="Defination of the network training learning rate.",
                        type=float)
    parser.add_argument("--logdir",default="log",help="Defination of logs file path.",type=str)
    parser.add_argument("--actfunc",default="sigmoid",help="Defination of activate function.",type=str)
    parser.add_argument("--layers",default="50,30,10",help="Defination of model of hidden layers.",type=str)
    parser.add_argument("--percentage",default="0.7,0.15,0.15",help="Defination of data separation .",type=str)
    parser.add_argument("--epoches",default=50,help="Defination of loop numbers.",type=int)
    parser.add_argument("--lambd",default=0.0,help="Regularation of the loss function.",type=float)
    parser.add_argument("--batchnormalize",default=False,help="Whether the network has batchnormalization.",type=bool)
    parser.add_argument("--batch",default=1,help="Batchfiy the data.",type=int)
    parser.add_argument("--type",default="Red",help="The training data type.",type=str)
    args = parser.parse_args()

    # logging defination
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
    log_dir = os.path.join(os.getcwd(),args.logdir,)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir,rq+".log")
    fh = logging.FileHandler(log_file,mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # load data and preparing data
    download(os.path.join(os.getcwd(),args.datadir))
    percentage = list(map(float,args.percentage.split(",")))
    if sum(percentage)!=1.0:
        raise ValueError("Error for value percentage: %f,%f,%f"%(percentage[0],percentage[1],percentage[2]))
    if args.type == "Red":
        DataSet = DataExtractor(os.path.join(args.datadir,"winequality-red.csv"),percentage,args.batch)
    elif args.type == "White":
        DataSet = DataExtractor(os.path.join(args.datadir,"winequality-white.csv"),percentage,args.batch)
    else:
        raise ModuleNotFoundError("Unknown model for %s"%str(args.type))
    # model preparing
    dim_list = list(map(int,args.layers.split(",")))
    dim_list = [11] + dim_list
    dim_list.append(1)
    # args.batchnormalize, args.learning_rate, args.lambd, name = rq
    model = DNNNet(dim_list,args.actfunc,name = rq)
    # training data
    train(model,DataSet,args)
    # test data
    test(model,DataSet)
if __name__ == "__main__":
    main()