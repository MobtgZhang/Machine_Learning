import numpy as np
import matplotlib.pyplot as plt
class POS:
    def __init__(self,interval,fly_max,tab = "max",iterMax = 1000,pos_num = 40,c1 = 2.8,c2 = 1.3,omiga=0.729,alpha = 0.8):
    	self.interval = interval
    	self.tab = tab.strip()
    	self.iterMax = iterMax
    	self.c1 = c1
    	self.c2 = c2
    	self.omiga = omiga
    	self.alpha = alpha
    	self.pos_num = pos_num
    	self.x_seeds = interval[0] + np.random.rand(self.pos_num)*(interval[1] - interval[0])

    	self.fly_max = fly_max
    	self.fly_velocity = np.random.rand(self.pos_num)*2
    	self.birds_position = interval[0] + np.random.rand(self.pos_num)*(interval[1] - interval[0])
    	self.x_solus = np.zeros(self.pos_num)
    def update_volocity(self,x_data):
    	self.fly_velocity = self.fly_velocity*self.omiga + self.c1*np.random.random()*(self.birds_position - x_data)+self.c2*np.random.random()*(self.x_solus - x_data)
    	for k in range(self.pos_num):
    		if self.fly_velocity[k] >self.fly_max:
    			self.fly_velocity[k] = self.fly_max
    		elif self.fly_velocity[k] <-self.fly_max:
    			self.fly_velocity[k] = -self.fly_max
    		else:
    			pass
    	delta_x = self.alpha*self.fly_velocity
    	x_data = self.adjust_value(x_data,delta_x)
    	return x_data
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
            x2 = self.update_volocity(x1)
            f2 = self.func(x2)
            delta_f = f2 - f1
            x1 = deal(x1,x2,delta_f)
    def deal_max(self,x1,x2,delta_f):
        for k in range(self.pos_num):
            if delta_f[k] > 0:
            	x1[k]=x2[k]
            	self.x_solus[k] = x2[k]
            else:x1[k]=x1[k]
        return x1
    def deal_min(self,x1,x2,delta_f):
        for k in range(self.pos_num):
            if delta_f[k] > 0:
            	x1[k]=x2[k]
            	self.x_solus[k] = x2[k]
            else:x1[k]=x1[k]
        return x1
    def adjust_value(self,x_in,delta_x):
        x_out = np.zeros(self.pos_num)
        for k in range(self.pos_num):
            if x_in[k]+delta_x[k] >= self.interval[0] and x_in[k] + delta_x[k] <= self.interval[1]:
                x_out[k] = x_in[k] + delta_x[k]
            else:
                x_out[k] = x_in[k] 
        return x_out
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
def test_func(x):
    y = np.sin(x*x)*(x*x-5*x)
    return y
def main():
    interval = [-5,5]
    fly_max = 800
    savefig = "C:\\Users\\asus\\Desktop\\projects\\POS.png"
    model = POS(interval,fly_max,'max')
    model.solve(test_func)
    model.display(savefig)
if __name__ == '__main__':
	main()