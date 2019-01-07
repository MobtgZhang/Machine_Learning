import numpy as np
import matplotlib.pyplot as plt
import random
class ACA:
    def __init__(self,interval,tab = "min",iterMax = 1000,ant_num = 40,alpha = 1,beta = 5,vol = 0.2,Q_value = 10):
        self.interval = interval
        self.tab = tab.strip()
        self.iterMax = iterMax
        self.alpha = alpha
        self.beta = beta
        self.vol = vol
        self.Q_value = Q_value
        self.ant_num = ant_num

        self.x_seeds = interval[0] + np.random.rand(self.ant_num)*(interval[1] - interval[0])
        # 计算距离矩阵,初始化距离矩阵
        self.distance = self.cal_distance(self.x_seeds)
        self.func = None

        self.tao_f = np.random.rand(self.ant_num)
        self.yita_f = np.random.random(self.ant_num)
        self.delta_tao = np.zeros(self.ant_num)
        self.delta_yita = np.zeros(self.ant_num)
        self.cumsum = np.zeros(self.ant_num)
        self.probability = np.ones(self.ant_num)
    def cal_distance(self,x_data):
    	distance = np.zeros((self.ant_num,self.ant_num))
    	for k in range(self.ant_num):
    		for j in range(self.ant_num):
    			delta = x_data[k] - x_data[j]
    			distance[k][j] = np.square(delta)
    	return distance
    def solve(self,func):
        self.func = func
        temp = "deal_" + self.tab
        if hasattr(self, temp):
            deal = getattr(self, temp)
        else:
            exit('>>>tab标签传参有误："min"|"max"<<<')
        x1 = self.x_seeds
        for k in range(self.iterMax):
            f1 = self.func(x1)
            delta_x = np.random.rand(self.ant_num)*2 - 1
            x2,delta_x = self.adjust_value(x1,delta_x)
            f2 = self.func(x2)
            delta_f = f2 - f1
            x1 = deal(x1,x2,delta_f)
            self.update_tao_f(delta_x)
            self.update_yita_f(delta_x)
            self.update_probability()
        self.x_solus = x1
    def update_probability(self):
        sum_value = 0
        for k in range(self.ant_num):
        	sum_value += np.power(self.delta_tao[k],self.alpha)*np.power(self.delta_yita[k],self.beta)
        for k in range(self.ant_num):
            if self.probability[k]<0.1:
                self.probability[k] = 0
            else:
                valueA = np.power(self.delta_tao[k],self.alpha)*np.power(self.delta_yita[k],self.beta)
                if (sum_value - valueA)<0.1:
                	self.probability[k] = 1
                else:	
                	self.probability[k] = valueA/sum_value
    def update_tao_f(self,delta_x):
        for k in range(self.ant_num):
            if self.cumsum[k]<0.1:
                self.delta_tao[k] = 0
            else:
                self.delta_tao[k] = self.Q_value/self.cumsum[k]
        self.cumsum = self.cumsum + delta_x
        self.tao_f = (1-self.vol)*self.tao_f + self.delta_tao
    def update_yita_f(self,delta_x):
        for k in range(self.ant_num):
            if self.cumsum[k]<0.1:
                self.delta_yita[k] = 0
            else:
                self.delta_yita[k] = self.Q_value/self.cumsum[k]
        self.cumsum = self.cumsum + delta_x
        self.yita_f = (1-self.vol)*self.yita_f + self.delta_yita
    def deal_max(self,x1,x2,delta_f):
        x = np.zeros(self.ant_num)
        for k in range(self.ant_num):
            if delta_f[k] > 0:x[k]=x2[k]
            else:x[k]=x1[k]
        return x
    def deal_min(self,x1,x2,delta_f):
        x = np.zeros(self.ant_num)
        for k in range(self.ant_num):
            if delta_f[k] < 0:x[k]=x2[k]
            else:x[k]=x1[k]
        return x
    def display(self,savefig):
        print('seed: {}\nsolution: {}'.format(self.x_seeds, self.x_solus))
        plt.figure(figsize = (6,4))
        x = np.linspace(self.interval[0], self.interval[1], 300)
        y = self.func(x)
        plt.plot(x, y, 'g-', label='function')
        plt.plot(self.x_seeds, self.func(self.x_seeds), 'bo', label='seed')
        plt.plot(self.x_solus, self.func(self.x_solus), 'r*', label='solution')
        plt.title('Display solutions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(savefig, dpi=500)
        plt.show()
        plt.close()
    def adjust_value(self,x_in,delta_x):
        x_out = np.zeros(self.ant_num)
        for k in range(self.ant_num):
            if x_in[k]+delta_x[k] >= self.interval[0] and x_in[k] + delta_x[k] <= self.interval[1]:
                x_out[k] = x_in[k] + delta_x[k]
            else:
                x_out[k] = x_in[k] - delta_x[k]
                delta_x[k] = -delta_x[k]
        return x_out,delta_x
def squre(x):
	return 1/np.power(x,2)
def abs_fun(x):
	return 1/np.abs(x)
def test_func(x):
    y = np.sin(x*x)*(x*x-5*x)
    return y
def main():
    interval = [-5,5]
    savefig = "C:\\Users\\asus\\Desktop\\projects\\ACA.png"
    model = ACA(interval,'max')
    model.solve(test_func)
    model.display(savefig)
if __name__ == '__main__':
	main()