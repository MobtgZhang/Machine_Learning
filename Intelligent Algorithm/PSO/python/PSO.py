import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import time
import math

from set import exchangeSeq,Sequence
def download(root_path,filename):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    if not os.path.exists(os.path.join(root_path,filename)):
        url = "http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/"+filename
        urllib.request.urlretrieve(url,os.path.join(root_path,filename))
        print("The data set: %s downloaded!"%os.path.join(root_path,filename))
    else:
        print("The data set: %s already has downloaded!"%os.path.join(root_path,filename))
def get_data(filename):
    data_list = []
    with open(filename,mode="r") as f:
        flag = False
        while True:
            line = f.readline()
            if "EOF" in line:
                break
            elif "NODE_COORD_SECTION" in line:
                flag = True
            elif flag:
                tmp = line.strip().split(" ")
                data_list.append([float(item) for item in tmp])
    return np.array(data_list)
class PSO:
    def __init__(self,cor_num,iterations,alpha,beta,omiga):
        self.cor_num = cor_num
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.omiga =omiga
        self.name = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
        self.citys_mat = None
    def solve(self,citys_mat):
        # 城市的数量
        citys_num = citys_mat.shape[0]
        self.citys_mat = citys_mat
        # 城市的距离矩阵计算方法
        citys_x = citys_mat[:, 0].reshape(citys_num, 1).dot(np.ones((1, citys_num)))
        citys_y = citys_mat[:, 1].reshape(citys_num, 1).dot(np.ones((1, citys_num)))
        citys_distance = np.sqrt(np.square(citys_x - citys_x.T) + np.square(citys_y - citys_y.T))
        # 初始化粒子群中的速度(交换序)和位置信息
        velocity_current = self.randseed(citys_num)
        sol_current = np.zeros((self.cor_num,citys_num),dtype=np.int)
        # 记录群体中粒子历史最好的长度信息
        self.sol_best_length = np.inf
        index = 0
        # 记录每个粒子的最好长度信息
        x_pos_best_length = np.zeros((self.cor_num,))
        for k in range(self.cor_num):
            sol_current[k] = np.random.permutation(citys_num)
            length_sum = self.cal_length(sol_current[k],citys_distance)
            x_pos_best_length[k] = length_sum
            if length_sum <self.sol_best_length:
                self.sol_best_length = length_sum
                index = k
        # 记录群体中粒子历史最好的路径信息
        self.sol_best_path = sol_current[index]
        # 记录每个粒子最好的位置信息
        x_pos_best_path = sol_current.copy()


        # 记录每一次迭代过程中总体粒子计算长度
        self.length_avg = np.zeros((self.iterations,))
        for iter in range(self.iterations):
            # 更新位置信息和速度信息
            # 当前群体中最好的序列(位置)
            pg_best = Sequence(self.sol_best_path)
            index_best = 0
            for k in range(self.cor_num):
                # 当前序列信息(位置)
                x_pos = Sequence(sol_current[k])
                # 当前的交换序列(速度)
                v_old = velocity_current[k]
                # 粒子历史最好的序列(位置)
                px_best = Sequence(x_pos_best_path[k])
                if np.random.rand()>(1-self.omiga):
                    v_new = self.randSetVelocity(pg_best-x_pos,self.beta)
                else:
                    v_new = v_old + self.randSetVelocity(px_best - x_pos, self.alpha) + self.randSetVelocity(
                        pg_best - x_pos, self.beta)
                x_pos = x_pos + v_new
                # 更新速度
                velocity_current[k] = v_new
                # 更新位置信息
                sol_current[k] = x_pos.sequence
                # 比较历史中各个粒子和群体最优值,然后更新粒子的长度信息和位置信息
                len_curr = self.cal_length(x_pos.sequence,citys_distance)
                if len_curr < x_pos_best_length[k]:
                    x_pos_best_length[k] = len_curr
                    x_pos_best_path[k] = x_pos.sequence
                # 更新全局信息
                if len_curr <self.sol_best_length:
                    self.sol_best_path = x_pos.sequence
                    self.sol_best_length = len_curr
                    index_best = k
            length_sum = 0
            for k in range(self.cor_num):
                length_sum = length_sum + self.cal_length(sol_current[k],citys_distance)
            self.length_avg[iter] = length_sum
            print("epoches:%d,length:%.4f,The best index:%d"%(iter,self.sol_best_length,index_best))
            # print(velocity_current[index_best])

    def randseed(self,length):
        '''
        产生随机交换子
        :param length:
        :return velocity:
        '''
        velocity = []
        for j in range(self.cor_num):
            tmp = exchangeSeq()
            for k in range(length):
                x_index = np.random.randint(0,length)
                y_index = np.random.randint(0,length)
                tmp.append((x_index,y_index))
            velocity.append(tmp)
        return velocity
    def cal_length(self,path_list,distance_mat):
        '''
        计算出具体的路径长度信息
        :param path_list:
        :param distance_mat:
        :return:
        '''
        citys_num = path_list.shape[0]
        length_sum = 0
        for k in range(citys_num-1):
            length_sum = length_sum + distance_mat[path_list[k],path_list[k+1]]
        length_sum = length_sum + distance_mat[path_list[-1], path_list[0]]
        return length_sum
    def draw(self):
        print(self.sol_best_path)
        print(self.sol_best_length)
        if not os.path.exists("log"):
            os.mkdir("log")
        # draw loss
        x = np.linspace(0, len(self.length_avg) - 1, len(self.length_avg))
        y = np.array(self.length_avg)
        plt.plot(x, y)
        plt.title(label="loss")
        plt.savefig(os.path.join("log", "%s_loss.png" % self.name))
        plt.close()

        # draw dots
        for k in range(0, len(self.sol_best_path) - 1):
            start = self.citys_mat[self.sol_best_path[k]]
            end = self.citys_mat[self.sol_best_path[k + 1]]
            plt.plot(start[0], start[1], "bo")
            plt.plot(end[0], end[1], "bo")
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                      length_includes_head=True, head_width=0.2, head_length=0.3, lw=1,
                      color="r")
        start = self.citys_mat[self.sol_best_path[-1]]
        end = self.citys_mat[self.sol_best_path[0]]
        plt.plot(start[0], start[1], "bo")
        plt.plot(end[0], end[1], "bo")
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  length_includes_head=True, head_width=0.2, head_length=0.3, lw=1,
                  color="r")
        plt.title(label="length:%.2f" % (self.sol_best_length))
        plt.savefig(os.path.join("log", "%s_route.png" % self.name))
        plt.show()
    def randSetVelocity(self,v_pre,per_rate):
        '''
        对于交换序列产生随机扰动
        产生随机扰动的方法有
        1. 随机进行两交换或者三交换
        2. 交换序列进行变异操作(丢弃或者增加操作，丢弃若干个元素或者是增加若干个元素)
        :param v_pre:
        :return:
        '''
        # 进行随机交换操作
        #print(len(v_pre))
        if np.random.rand() > (1-per_rate) and len(v_pre)>1:
            #print(len(v_pre))
            # 进行两两交换操作
            v_next = v_pre.copy()
            if np.random.rand() > 0.5:
                length = len(v_next)
                ind1 =0;ind2 = 0
                while ind1 == ind2:
                    ind1 = np.random.randint(0,length)
                    ind2 = np.random.randint(0,length)
                v_next.exchange_list[ind1], v_next.exchange_list[ind2] = v_next.exchange_list[ind2], v_next.exchange_list[ind1]
                return v_next
            else:
                # 进行算子的变异操作
                v_next = v_pre.copy()
                length = v_next.max()
                if np.random.rand()>0.5:
                    x_index =0;y_index = 0
                    while (x_index == y_index):
                        x_index = np.random.randint(0,length)
                        y_index = np.random.randint(0,length)
                    v_next.append((x_index,y_index))
                else:
                    if np.random.rand()<=0.5 and len(v_next)>1:
                        v_next.exchange_list.pop()
                    if np.random.rand()*np.random.rand()>0.5:
                        len_v = np.random.randint(0,length)
                        for j in range(len_v):
                            x_index = 0;y_index = 0
                            while (x_index == y_index):
                                x_index = np.random.randint(0, length)
                                y_index = np.random.randint(0, length)
                            v_next.append((x_index, y_index))
                return v_next
        else:
            return v_pre
def main():
    filename = "eil51.tsp"
    root_path = "data"
    download(root_path,filename)
    data_list = get_data(os.path.join(root_path,filename))
    cor_num = 300
    # 变异率
    omiga = 0.2
    alpha = 0.7
    beta = 0.4

    epoches = 400
    model = PSO(cor_num,epoches,alpha,beta,omiga)
    model.solve(data_list[:,1:])
    model.draw()
if __name__ == '__main__':
    main()