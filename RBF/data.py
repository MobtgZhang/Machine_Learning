import urllib.error
import urllib.request
import logging
import os
import csv
import numpy as np
import math
def download(root_dir):
    logger = logging.getLogger()
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    regression_path = os.path.join(root_dir,"regression")
    if not os.path.exists(regression_path):
        os.mkdir(regression_path)
    classification_path = os.path.join(root_dir,"classification")
    if not os.path.exists(classification_path):
        os.mkdir(classification_path)
    regression_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00514/"
    regression_file = "Bias_correction_ucl.csv"
    file_dir = os.path.join(regression_path,regression_file)
    if not os.path.exists(file_dir):
        urllib.request.urlretrieve(regression_url + regression_file, file_dir)
        logger.info(file_dir + " downloaded!")
    else:
        logger.info(file_dir + " has downloaded!")
    classification_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    classification_namefiles = ["winequality-red.csv", "winequality-white.csv", "winequality.names"]
    for k in range(len(classification_namefiles)):
        path_dir = os.path.join(classification_path, classification_namefiles[k])
        file_url = classification_url + classification_namefiles[k]
        if not os.path.exists(path_dir):
            urllib.request.urlretrieve(file_url, path_dir)
            logger.info(path_dir + " downloaded!")
        else:
            logger.info(path_dir + " has downloaded!")
class DataExtractorWine:
    def __init__(self,datadir,percentage,batch):
        self.datadir = datadir
        self.batch = batch
        self.headers = None
        self.percentage = percentage
        self.load_data()
    def load_data(self):
        stored_data = []
        with open(self.datadir) as f:
            reader = csv.reader(f,delimiter=";")
            flag = True
            num = 0
            for row in reader:
                num+=1
                if flag:
                    self.headers = row
                    flag = False
                else:
                    tmp_data = [float(item) for item in row]
                    stored_data.append(tmp_data)
        stored_data = np.array(stored_data)
        length = len(stored_data)
        train_index = int(self.percentage[0] * length)
        valid_index = int((self.percentage[0] + self.percentage[1]) * length)
        np.random.shuffle(stored_data)
        self.stored_data_trian = stored_data[:train_index, :]
        self.stored_data_valid = stored_data[train_index:valid_index, :]
        self.stored_data_test = stored_data[valid_index:, :]

    def getitem(self, item, index):
        if index == 0:
            # 将最后一个数值变为one-hot向量
            num = self.stored_data_trian[item * self.batch:item * self.batch + self.batch, -1]
            length = len(num)
            result = np.zeros((length,10))
            for k in range(length):
                result[k,int(num[k])] = 1
            return self.stored_data_trian[item * self.batch:item * self.batch + self.batch,:-1], result
        elif index == 1:
            num = self.stored_data_valid[item * self.batch:item * self.batch + self.batch, -1]
            length = len(num)
            result = np.zeros((length, 10))
            for k in range(length):
                result[k, int(num[k])] = 1
            return self.stored_data_valid[item * self.batch:item * self.batch + self.batch,:-1], result
        elif index == 2:
            num = self.stored_data_test[item * self.batch:item * self.batch + self.batch, -1]
            length = len(num)
            result = np.zeros((length, 10))
            for k in range(length):
                result[k, int(num[k])] = 1
            return self.stored_data_test[item * self.batch:item * self.batch + self.batch, :-1], result
        else:
            raise IndexError("Error for index number:%d ,index number must be 0,1,2." % index)
    def getvalidate(self):
        num = self.stored_data_valid[:, -1]
        length = len(num)
        result = np.zeros((length, 10))
        for k in range(length):
            result[k, int(num[k])] = 1
        return self.stored_data_valid[:, :-1],result
    def gettest(self):
        num = self.stored_data_valid[:, -1]
        length = len(num)
        result = np.zeros((length, 10))
        for k in range(length):
            result[k, int(num[k])] = 1
        return self.stored_data_test[:, :-1],result
    def getlen(self, index):
        if index == 0:
            return math.ceil(self.stored_data_trian.shape[0] / self.batch)
        elif index == 1:
            return math.ceil(self.stored_data_valid.shape[0] / self.batch)
        elif index == 2:
            return math.ceil(self.stored_data_test.shape[0] / self.batch)
        else:
            raise IndexError("Error for index number:%d ,index number must be 0,1,2." % index)
class DataExtractorForecast:
    def __init__(self,datadir,percentage,batch,):
        self.datadir = datadir
        self.batch = batch
        self.headers = None
        self.percentage = percentage
        self.load_data()
    def load_data(self):
        stored_data = []
        with open(self.datadir,mode="r") as f:
            reader = csv.reader(f,delimiter=",")
            flag = True
            num = 0
            for row in reader:
                if flag:
                    self.headers = row
                    flag = False
                else:
                    # 去掉NanA数据
                    if "NaN" in row:
                        continue
                    else:
                        # 去掉第一列数据和第二列数据
                        tmp_data = [float(row[k]) for k in range(2,len(row))]
                        stored_data.append(tmp_data)
        stored_data = np.mat(stored_data)
        length = len(stored_data)
        train_index = int(self.percentage[0] * length)
        valid_index = int((self.percentage[0] + self.percentage[1]) * length)
        np.random.shuffle(stored_data)
        self.stored_data_trian = stored_data[:train_index, :]
        self.stored_data_valid = stored_data[train_index:valid_index, :]
        self.stored_data_test = stored_data[valid_index:, :]
    def getitem(self, item, index):
        if index == 0:
            return self.stored_data_trian[item * self.batch:item * self.batch + self.batch,:-2], \
                   self.stored_data_trian[item * self.batch:item * self.batch + self.batch, -2:]
        elif index == 1:
            return self.stored_data_valid[item * self.batch:item * self.batch + self.batch,:-2], \
                   self.stored_data_valid[item * self.batch:item * self.batch + self.batch, -2:]
        elif index == 2:
            return self.stored_data_test[item * self.batch:item * self.batch + self.batch, :-2], \
                   self.stored_data_test[item * self.batch:item * self.batch + self.batch,-2:]
        else:
            raise IndexError("Error for index number:%d ,index number must be 0,1,2." % index)
    def getvalidate(self):
        return self.stored_data_valid[:, :-2], self.stored_data_valid[:, -2:]
    def gettest(self):
        return self.stored_data_test[:, :-2], self.stored_data_test[:, -2:]
    def getlen(self, index):
        if index == 0:
            return math.ceil(self.stored_data_trian.shape[0] / self.batch)
        elif index == 1:
            return math.ceil(self.stored_data_valid.shape[0] / self.batch)
        elif index == 2:
            return math.ceil(self.stored_data_test.shape[0] / self.batch)
        else:
            raise IndexError("Error for index number:%d ,index number must be 0,1,2." % index)
if __name__ == '__main__':
    datadir = "K:\\PycharmProjects\\RBF\data\\regression\\Bias_correction_ucl.csv"
    DataSet = DataExtractorForecast(datadir=datadir,percentage=[0.7,0.15,0.15],batch=5)
    a,b = DataSet.getitem(0,2)
    print(a.shape,b.shape)