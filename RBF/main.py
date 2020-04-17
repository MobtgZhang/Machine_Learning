import os
import logging
import argparse
import time

from data import download
from data import DataExtractorForecast,DataExtractorWine
from model import RBFBPRegression,RBFBPClassification,RBFGradClassification,RBFGradRegression,RBFClassification,RBFRegression
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def train(model,loadedData,args):
    logger = logging.getLogger()
    logger.info("Training begining for dataset:%s dataset"%args.type)
    for k in range(args.epoches):
        loss_sum = 0
        length = loadedData.getlen(0)
        for j in range(length):
            input,target = loadedData.getitem(j,0)
            model.backward(input,target,args.learning_rate,args.lambd)
            output = model.forward(input)
            loss = model.loss(output,target)
            loss_sum += loss
        loss = loss_sum/length
        logger.info("Training loss value:%f" % (loss))
        # validate the result
        input, target = loadedData.getvalidate()
        output = model.forward(input)
        loss = model.loss(output, target)
        logger.info("Validating loss value:%f" % (loss))
    logger.info("Trained  for dataset:%s dataset completed." % args.type)
    # save data parameters

def test(model,loadedData,args):
    logger = logging.getLogger()
    input,target = loadedData.gettest()
    output = model.forward(input)
    loss = model.loss(output,target)
    logger.info("Test loss value:%f" % (loss))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",default="data",help="Defination of the data file path",type=str)
    parser.add_argument("--logdir", default="log", help="Defination of logs file path.", type=str)
    parser.add_argument("--type",default="Red-Wine",help="Defination of the problem",type=str)
    parser.add_argument("--percentage", default="0.7,0.15,0.15", help="Defination of data separation .", type=str)
    parser.add_argument("--batch", default=5, help="Batchfiy the data.", type=int)
    parser.add_argument("--model",default="RBFBP",help="Defination of the model",type=str)
    parser.add_argument("--act_func",default="Gauss",help="Defination of the kernel activate function.",type=str)
    parser.add_argument("--epoches",default=50,help="Defination of loop numbers.",type=int)
    parser.add_argument("--learning_rate", default=0.3, help="Defination of the network training learning rate.",type=float)
    parser.add_argument("--lambd", default=0.1, help="Regularation of the loss function.", type=float)
    parser.add_argument("--hid-dim", default=150, help="Defination of RBFBP model hidden layers.", type=int)
    args =parser.parse_args()
    # preparing log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    log_dir = os.path.join(os.getcwd(), args.logdir,)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, rq + ".log")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # download dataset
    download(os.path.join(os.getcwd(), args.datadir))
    percentage = list(map(float, args.percentage.split(",")))
    if sum(percentage) != 1.0:
        raise ValueError("Error for value percentage: %f,%f,%f" % (percentage[0], percentage[1], percentage[2]))
    # preparing dataset
    if args.type == "Red-Wine":
        DataSet = DataExtractorWine(os.path.join(args.datadir,"classification","winequality-red.csv"),percentage,args.batch)
    elif args.type == "White-Wine":
        DataSet = DataExtractorWine(os.path.join(args.datadir,"classification","winequality-white.csv"),percentage,args.batch)
    elif args.type == "Forecast":
        DataSet = DataExtractorForecast(os.path.join(args.datadir,"regression","Bias_correction_ucl.csv"),percentage,args.batch)
    else:
        raise ValueError("Error for the dataset:%s"%args.type)
    # preparing model
    if args.model == "RBFBP" and (args.type == "Red-Wine" or args.type == "White-Wine"):
        model = RBFBPClassification(in_dim=11,hid_dim=args.hid_dim,n_class=10,act_name=args.act_func,name=rq)
    elif args.model == "RBFGrad" and (args.type == "Red-Wine" or args.type == "White-Wine"):
        model = RBFGradClassification(in_dim=11,n_class=10,act_name=args.act_func,name=rq)
    elif args.model == "RBF" and (args.type == "Red-Wine" or args.type == "White-Wine"):
        model = RBFClassification(in_dim=11,n_class=10,act_name=args.act_func,name=rq)
    elif args.model == "RBFBP" and args.type == "Forecast":
        model = RBFBPRegression(in_dim=21,hid_dim=args.hid_dim,out_dim=2,act_name=args.act_func,name=rq)
    elif args.model == "RBFGrad" and args.type == "Forecast":
        model = RBFGradRegression(in_dim=21,out_dim=2,act_name=args.act_func,name=rq)
    elif args.model == "RBF" and args.type == "Forecast":
        model = RBFRegression(in_dim=21,out_dim=2,act_name=args.act_func,name=rq)
    else:
        raise TypeError("Unknow model:%s"%args.model)
    # training data
    train(model,DataSet,args)
    # testing data
    test(model,DataSet,args)
if __name__ == '__main__':
    main()