import urllib.error
import urllib.request
import os
import csv
import logging
import numpy as np
import math
def download(root_dir):
    attachment_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    namefiles= ["winequality-red.csv","winequality-white.csv","winequality.names"]
    logger = logging.getLogger()
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    for k in range(len(namefiles)):
        path_dir = os.path.join(root_dir,namefiles[k])
        file_url = attachment_url+namefiles[k]
        if not os.path.exists(path_dir):
            try:
                urllib.request.urlretrieve(file_url,path_dir)
                logger.info(path_dir + " downloaded!")
            except urllib.error.URLError:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                urllib.request.urlretrieve(file_url,path_dir)
                logger.info(path_dir + " downloaded!")
        else:
            logger.info("Data "+namefiles[k]+" has downloaded!")
class DataExtractor:
    def __init__(self,datadir,percentage,batch):
        self.datadir = datadir
        self.batch = batch
        self.headers = None
        self.stored_data_trian = []
        self.stored_data_valid = []
        self.stored_data_test = []
        self.percentage = percentage
        self.load_data()
    def load_data(self):
        stored_data = []
        with open(self.datadir) as f:
            reader = csv.reader(f,delimiter=";")
            flag = True
            for row in reader:
                if flag:
                    self.headers = row
                    flag = False
                else:
                    temp_data = [float(item) for item in row]
                    stored_data.append(temp_data)
        stored_data = np.mat(stored_data)
        length = len(stored_data)
        train_index = int(self.percentage[0]*length)
        valid_index = int((self.percentage[0]+self.percentage[1])*length)
        np.random.shuffle(stored_data)
        self.stored_data_trian = stored_data[:train_index,:]
        self.stored_data_valid = stored_data[train_index:valid_index,:]
        self.stored_data_test = stored_data[valid_index:,:]
    def getitem(self, item,index):
        if index == 0:
            return self.stored_data_trian[item*self.batch:item*self.batch+self.batch,:-1],self.stored_data_trian[item*self.batch:item*self.batch+self.batch,-1]
        elif index==1:
            return self.stored_data_valid[item*self.batch:item*self.batch+self.batch,:-1],self.stored_data_valid[item*self.batch:item*self.batch+self.batch,-1]
        elif index==2:
            return self.stored_data_test[item*self.batch:item*self.batch+self.batch,:-1], self.stored_data_test[item*self.batch:item*self.batch+self.batch,-1]
        else:
            raise IndexError("Error for index number:%d ,index number must be 0,1,2."%index)
    def getvalidate(self):
        return self.stored_data_valid[:,:-1],self.stored_data_valid[:,-1]
    def gettest(self):
        return self.stored_data_test[:,:-1],self.stored_data_test[:,-1]
    def getlen(self,index):
        if index==0:
            return math.ceil(self.stored_data_trian.shape[0]/self.batch)
        elif index==1:
            return math.ceil(self.stored_data_valid.shape[0]/self.batch)
        elif index ==2:
            return math.ceil(self.stored_data_test.shape[0]/self.batch)
        else:
            raise IndexError("Error for index number:%d ,index number must be 0,1,2." % index)