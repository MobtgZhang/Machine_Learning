import numpy as np
import matplotlib.pyplot as plt
# 使用浮点数进行编码
class GA:
    def __init__(self,popsize,pc,pm,iterations,alpha=0.9):
        self.popsize = popsize
        self.pc = pc
        self.pm = pm
        self.alpha = alpha
        self.iterations = iterations
    def solve(self,func,x_min,x_max):
        # 初始化种群
        pop_current = x_min + np.random.random(self.popsize,)*(x_max-x_min)
        self.draw(func,pop_current,x_min,x_max,"pic1.png")
        for _iter in range(self.iterations):
            #计算种群的适应度
            fit_val = func(pop_current)
            # 随机选择两个算子进行交叉结合
            if np.random.random()<self.pc:
                index1, index2 = np.random.choice(self.popsize, 2)
                child1,child2 = self.crossover(pop_current[index1],pop_current[index2])
                new = sorted([child1,child2,pop_current[index1],pop_current[index2]],key=func,reverse=True)
                pop_current[index1] = new[0]
                pop_current[index2] = new[1]
            # 进行变异的操作
            if np.random.rand() < self.pm:
                index = np.random.choice(self.popsize)
                pop_current[index] = self.mutate(x_min,x_max)
        self.draw(func, pop_current, x_min, x_max, "pic2.png")
    def crossover(self,x_t,y_t):
        nx_t = self.alpha*x_t + (1-self.alpha)*y_t
        ny_t = self.alpha*y_t + (1-self.alpha)*x_t
        return nx_t,ny_t
    def mutate(self,x_min,x_max):
        return np.random.uniform(x_min,x_max)
    def draw(self,func,pop_current,x_min,x_max,filename):
        x = np.linspace(x_min,x_max,1000)
        y = func(x)
        plt.plot(x, y)
        plt.scatter(pop_current, func(pop_current), c="r")
        plt.grid()
        plt.savefig(filename)
        plt.close()
def func(x):
    return x+10*np.sin(5*x)+7*np.cos(4*x)
if __name__ == '__main__':
    popsize = 20
    pc = 0.6
    pm = 0.01
    iterations = 500
    model = GA(popsize,pc,pm,iterations)
    x_min = -10
    x_max = 10
    model.solve(func,x_min,x_max)
