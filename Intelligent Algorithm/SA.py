import numpy as np
import matplotlib.pyplot as plt
import random
class SA:
    def __init__(self,interval,tab = "min",T_max = 10000,T_min = 1,iterMax = 1000,rate = 0.95):
        self.interval = interval        # Given state space as the undetermined solution space. 
        self.T_max = T_max              # The initial temperature,upper limit of temperature .
        self.T_min = T_min              # Cut-off annealing temperature,lower temperature limit. 
        self.iterMax = iterMax          # The number of iterations within a constant temperature. 
        self.rate = rate                #  Annealing cooling rate
        self.x_seed = random.uniform(interval[0],interval[1])
        self.tab = tab.strip()
        self.func = None
    def solve(self,func):
        self.func = func
        temp = "deal_" + self.tab
        if hasattr(self, temp):
            deal = getattr(self, temp)
        else:
            exit('>>>tab标签传参有误："min"|"max"<<<')
        x1 = self.x_seed
        T = self.T_max
        while T>=self.T_min:
            for k in range(self.iterMax):
                f1 = self.func(x1)
                delta_x = random.random()*2 - 1
                if x1+delta_x >= self.interval[0] and x1 + delta_x <= self.interval[1]:
                    x2 = x1 + delta_x
                else:
                    x2 = x1 - delta_x
                f2 = self.func(x2)
                delta_f = f2 - f1
                x1 = deal(x1,x2,delta_f,T)
            T = T*self.rate
        self.x_solu = x1
    def p_min(self,deltax,T):
        probability = np.exp(-delta/T)
        return probability
    def p_max(self,delta,T):
        probability = np.exp(delta/T)
        return probability
    def deal_min(self,x1,x2,delta,T):
        if delta <0:
            return x2
        else:
            p = self.p_min(delta,T)
            if p>random.random():return x2
            else:return x1
    def deal_max(self,x1,x2,delta,T):
        if delta > 0:
            return x2
        else:
            p = self.p_max(delta,T)
            if p>random.random():return x2
            else:return x1
    def display(self,savefig):
        print('seed: {}\nsolution: {}'.format(self.x_seed, self.x_solu))
        plt.figure(figsize = (6,4))
        x = np.linspace(self.interval[0], self.interval[1], 300)
        y = self.func(x)
        plt.plot(x, y, 'g-', label='function')
        plt.plot(self.x_seed, self.func(self.x_seed), 'bo', label='seed')
        plt.plot(self.x_solu, self.func(self.x_solu), 'r*', label='solution')
        plt.title('solution = {}'.format(self.x_solu))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(savefig, dpi=500)
        plt.show()
        plt.close()
def test_func(x):
    y = np.sin(x*x)*(x*x-5*x)
    return y
if __name__ == '__main__':
    interval = [-5,5]
    savefig = "C:\\Users\\asus\\Desktop\\projects\\SA.png"
    model = SA(interval, 'max')
    model.solve(test_func)
    model.display(savefig)