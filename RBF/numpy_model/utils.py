import numpy as np
def select_func(name):
    if name == "Gauss":
        return Gauss
    elif name == "ReflextedSigmoid":
        return ReflextedSigmoid
    elif name == "InverseMulti":
        return InverseMulti
    else:
        raise TypeError("Unknow model:%s"%str(name))
class Gauss:
    def __init__(self):
        self.name = "Gauss"
    def forward(self,r,delta):
        return np.exp(-np.square(r)/(2*np.square(delta)))
    def diff(self,r,delta):
        return -np.multiply(np.divide(r,np.square(delta)), self.forward(r, delta))
class ReflextedSigmoid:
    def __init__(self):
        self.name = "ReflextedSigmoid"
    def forward(self,r,delta):
        return 1 / (1 + np.exp(np.square(r) / (np.square(delta))))
    def diff(self,r,delta):
        return self.forward(r, delta) * (self.forward(r, delta) - 1) * (2 * r / np.square(delta))
class InverseMulti:
    def __init__(self):
        self.name = "InverseMulti"
    def forward(self,r,delta):
        return 1/np.sqrt(np.square(r)+np.square(delta))
    def diff(self,r,delta):
        return (-r)*np.power(self.forward(r,delta),3)
class SoftMax:
    def __init__(self):
        self.name = "SoftMax"
    def forward(self,input,axis=0):
        if input.ndim == 1:
            return np.exp(input)/np.sum(np.exp(input))
        elif input.ndim == 2:
            output = input.copy()
            if axis == 0:
                for k in range(input.shape[0]):
                    output[k,:] = np.exp(input[k,:])/np.sum(np.exp(input[k,:]))
            elif axis == 1:
                for k in range(input.shape[1]):
                    output[:,k] = np.exp(input[:,k]) / np.sum(np.exp(input[:,k]))
            else:
                raise ValueError("Value of axis must be 0 or 1.")
            return output
        else:
            raise ValueError("Matrix size don't match:size must be 1 or 2.")