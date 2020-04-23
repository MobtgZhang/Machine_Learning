import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import time
def download(filename):
    url = "http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/"+filename
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url,filename)
    else:
        print("The file %s has downloaded!"%filename)
def getdata(filename):
    data_list = []
    with open(filename) as f:
        flag = False
        while True:
            line = f.readline()
            if "EOF" in line:
                break
            elif "NODE_COORD_SECTION" in line:
                flag = True
            elif flag:
                str_list = line.strip().split(" ")
                data_list.append([float(str_list[1]),float(str_list[2])])
            else:
                continue
    return np.array(data_list)
class GA:
    def __init__(self,popsize,iterations,pc=0.6,pc_num=0.4,pm = 0.05,pr =0.05):
        self.popsize = popsize  #种群的大小
        self.iterations = iterations #进化迭代的次数
        self.pc_num = pc_num # 确定交配的比例
        self.pc = pc # 交配的概率
        self.pm = pm # 变异的概率
        self.pr = pr # 倒位的概率
        self.citys_mat = None
        self.sol_best = None
        self.E_best = np.inf
        self.sol_length_avg = None
        self.pop_num = None
        self.name = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
    def solve(self,citys_mat):
        self.citys_mat = citys_mat
        citys_num = citys_mat.shape[0]
        # 计算城市的距离信息
        # 获取邻接矩阵
        citys_x = citys_mat[:, 0].reshape(citys_num, 1).dot(np.ones((1, citys_num)))
        citys_y = citys_mat[:, 1].reshape(citys_num, 1).dot(np.ones((1, citys_num)))
        citys_distance = np.sqrt(np.square(citys_x - citys_x.T) + np.square(citys_y - citys_y.T))
        # 初始化种群，生成随机的染色体
        pop_store = []
        for k in range(self.popsize):
            pop_store.append(np.random.permutation(citys_num))

        # 用于保存总体路径长度的变化
        self.sol_length_avg = []
        # 用于保存每一个批次的种群数量变化
        self.pop_num = [self.popsize]
        # 保存最好的路径长度以及路径信息
        # 注意种群的数量是可以变化的
        pop_size_current = self.popsize



        pc_prob = np.random.random() if self.pc_num is None else self.pc_num
        for _iter in range(self.iterations):
            # 用于保存当前的个体适应度的值
            fit_current = []
            # 计算种群的适应度
            fit_sum = 0
            for k in range(pop_size_current):
                fit_val = self.fitness(pop_store[k],citys_distance)
                fit_current.append(fit_val)
                fit_sum += fit_val
            # 进行自然选择操作,产生新的种群
            pop_store_new = []
            # 方法一：使用轮盘赌方法选择个体并产生新的种群

            # 用于保存当前染色体被复制的概率
            c_prob_current = [fit_current[k]/fit_sum for k in range(pop_size_current)]
            # 用于计算当前染色体被复制的累积概率值
            cum_prob_current = []
            prob_sum = 0
            for k in range(pop_size_current):
                prob_sum += c_prob_current[k]
                cum_prob_current.append(prob_sum)
            cum_prob_current = np.array(cum_prob_current)
            for k in range(pop_size_current):
                probability = np.random.random()
                index = np.where(cum_prob_current > probability)[0][0]
                pop_store_new.append(pop_store[index])
            # 更新种群信息
            pop_store = pop_store_new
            pop_size_current = len(pop_store)
            # 种群进行交配、个体进行变异、倒位
            # 交配过程
            # 确定交配染色体的数量
            number = int(pop_size_current * pc_prob)
            # 每50个循环使得交配概率下降一半
            if _iter%100 == 0:
                pc_prob = pc_prob/2
            for k in range(number):
                # 方法一：选择概率最小的两个个体进行交配
                # 产生pop_size 的随机数
                probability = np.random.rand(pop_size_current)
                tmp = np.sort(probability, kind="quicksort")
                num1 = tmp[0]
                num2 = tmp[1]
                index1 = np.where(probability == num1)[0][0]
                index2 = np.where(probability == num2)[0][0]
                # 选择两个进行交配
                new_itema, new_itemb = self.intercross(pop_store[index1], pop_store[index2], self.pc)
                if new_itemb is not None and new_itemb is not None:
                    pop_store.append(new_itema)
                    pop_store.append(new_itemb)
            # 每次交配完成之后,更新种群数量
            pop_size_current = len(pop_store)
            # 个体的基因突变和倒位
            for k in range(pop_size_current):
                pop_store[k] = GA.mutate(pop_store[k],self.pm)
                pop_store[k] = GA.reverse(pop_store[k],self.pr)
            self.pop_num.append(pop_size_current)
            # 计算最好的路径信息和平均路径信息变化情况
            length_sum = 0
            for k in range(pop_size_current):
                length =self.cal_distance(pop_store[k],citys_distance)
                if length < self.E_best:
                    self.E_best = length
                    self.sol_best = pop_store[k]
                length_sum += length
            self.sol_length_avg.append(length_sum/pop_size_current)
            print("iter:%d,pop_size:%d,best:%.4f"%(_iter,pop_size_current,self.E_best))
    def fitness(self,path,citys_distance):
        '''
        计算个体的适应度，适应度越大则容易存活下去
        :param path: 个体的染色体
        :return:
        '''
        length_sum = self.cal_distance(path,citys_distance)
        return 1.0/length_sum
    @staticmethod
    def intercross(item_a,item_b,rate):
        '''
        表示染色体之间交叉的操作
        :param item_a: 个体染色体a
        :param item_b: 个体染色体b
        :param rate: 交叉的概率
        :return:
        '''
        if np.random.random() <rate:
            new_itema = item_a.copy()
            new_itemb = item_b.copy()
            citys_num = new_itema.shape[0]
            r1 = 0;r2=0
            while (r1 == r2):
                r1 = np.random.randint(0,citys_num)
                r2 = np.random.randint(0,citys_num)

            start = min(r1,r2)
            end = max(r1,r2) + 1
            for k in range(start, end):
                new_itema[k],new_itemb[k] = new_itemb[k],new_itema[k]
            # 建立映射关系表
            set_a = set()
            set_b = set()
            for k in range(citys_num):
                for j in range(k+1,citys_num):
                    if new_itema[k] == new_itema[j]:
                        if j not in range(start,end):
                            set_a.add(j)
                        if k not in range(start,end):
                            set_a.add(k)
                    if new_itemb[k] == new_itemb[j]:
                        if j not in range(start,end):
                            set_b.add(j)
                        if k not in range(start,end):
                            set_b.add(k)
            for i,j in zip(set_a,set_b):
                new_itema[i],new_itemb[j] = new_itemb[j],new_itema[i]
            return new_itema,new_itemb
        else:
            return None,None
    @staticmethod
    def mutate(pop_item,rate):
        '''
        表示一个个体的染色体的变异操作
        :param select_item: 染色体
        :param rate: 变异的概率
        :return:
        '''
        new_item = pop_item.copy()
        if np.random.random() < rate:
            citys_num = pop_item.shape[0]
            index1 = np.random.randint(0,citys_num)
            index2 = np.random.randint(0,citys_num)
            new_item[index1],new_item[index2] =new_item[index2],new_item[index1]
        return new_item
    @staticmethod
    def reverse(pop_item,rate):
        '''
        表示染色体的倒位
        :param pop_item:个体染色体序列
        :param start: 开始倒位的位置
        :param end: 结束倒位的位置
        :param rate: 发生倒位的概率
        :return:
        '''
        new_item = pop_item.copy()
        if np.random.random()<rate:
            citys_num = pop_item.shape[0]
            start = np.random.randint(0,citys_num)
            end = np.random.randint(0,citys_num)
            start = min(start,end)
            end = max(start,end)
            tmp = new_item[start:end]
            tmp_len = len(tmp)
            for k in range(tmp_len//2):
                tmp[k],tmp[tmp_len-k-1] = tmp[tmp_len-k-1],tmp[k]
        return new_item
    def cal_distance(self,path,citys_distance):
        '''
        用于计算路径的长度信息
        :param path:
        :param citys_distance:
        :return:
        '''
        citys_num = path.shape[0]
        length_sum = 0
        #print(type(path[0]))
        for k in range(citys_num-1):
            length_sum = length_sum + citys_distance[path[k],path[k+1]]
        length_sum = length_sum + citys_distance[path[-1],path[0]]
        return length_sum
    def draw(self):
        print(self.sol_best)
        print(self.E_best)
        if not os.path.exists("log"):
            os.mkdir("log")
        # draw loss
        x = np.linspace(0, len(self.sol_length_avg) - 1, len(self.sol_length_avg))
        y = np.array(self.sol_length_avg)
        plt.plot(x, y)
        plt.title(label="loss")
        plt.savefig(os.path.join("log", "%s_loss.png" % self.name))
        plt.close()
        # draw pop_size
        x = np.linspace(0,self.iterations,self.iterations+1)
        y = np.array(self.pop_num)
        plt.plot(x,y)
        plt.xlabel("iterations")
        plt.ylabel("numbers")
        plt.savefig(os.path.join("log", "%s_pop_size.png" % self.name))
        plt.close()
        # draw dots
        for k in range(0, len(self.sol_best) - 1):
            start = self.citys_mat[self.sol_best[k]]
            end = self.citys_mat[self.sol_best[k + 1]]
            plt.plot(start[0], start[1], "bo")
            plt.plot(end[0], end[1], "bo")
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                      length_includes_head=True, head_width=0.2, head_length=0.3, lw=1,
                      color="r")
        start = self.citys_mat[self.sol_best[-1]]
        end = self.citys_mat[self.sol_best[0]]
        plt.plot(start[0], start[1], "bo")
        plt.plot(end[0], end[1], "bo")
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  length_includes_head=True, head_width=0.2, head_length=0.3, lw=1,
                  color="r")
        plt.title(label="length:%.2f" % self.E_best)
        plt.savefig(os.path.join("log", "%s_route.png" % self.name))
        plt.show()
def main():
    filename = "eil51.tsp"
    download(filename)
    data_list = getdata(filename)
    popsize = 50
    iterations = 2000
    pc = 0.4
    pc_num = 0.06
    pm = 0.2
    pr = 0.3
    model = GA(popsize,iterations,pc,pc_num,pm,pr)
    model.solve(data_list[1:])
    model.draw()
if __name__ == '__main__':
    main()