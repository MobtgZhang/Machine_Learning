import theano
import theano.tensor as T
from model import Module
class SGD:
    def __init__(self,learning_rate,lambd):
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.step = None
    def initial(self,inputs,model):
        grad_list = []
        for name, value in vars(model).items():
            if isinstance(value, Module):
                parameters = value.parameters()
                gradients = value.parameters()
                for param in parameters:
                    grad_list.append([parameters[param], parameters[param] - self.learning_rate * (
                                gradients[param] + self.lambd * parameters[param])])
        parameters = model.parameters()
        gradients = model.parameters()
        for param in parameters:
            grad_list.append([parameters[param], parameters[param] - self.learning_rate * (
                    gradients[param] + self.lambd * parameters[param])])
        self.step = theano.function(inputs=inputs,outputs=None,updates=grad_list)
