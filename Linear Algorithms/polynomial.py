import numpy as np
from model import cholesky_LDLT_solve
class Polynomial:
    def __init__(self,degree,lambd=0):
        self.degree = degree
        self.lambd = lambd
        self.Mat = None
        self.Bais = None
        self.weight = None
    def fit(self,input,target):
        if input.shape != target.shape:
            raise ValueError("size of matrix doesn't match!")
        degree = self.degree
        self.Mat = np.zeros((degree,degree))
        self.Bais = np.zeros(degree)
        for k in range(degree):
            for j in range(degree):
                self.Mat[k,j] = np.sum(np.power(input,k+j))
        for j in range(degree):
            self.Mat[k,j] -= self.lambd/2
            self.Bais[j] = np.sum(np.power(input,j)*target)
        self.weight = cholesky_LDLT_solve(self.Mat,self.Bais)
    def regress(self,input):
        if type(input)==np.ndarray:
            if input.ndim !=1:
                raise ValueError("size of matrix doesn't match!")
            output = np.zeros(input.shape)
            for i in range(len(input)):
                output[i] = np.sum(np.power(input[i],np.arange(self.degree))*self.weight)
            return output
        elif type(input) == float or type(input) == int:
            output = np.sum(np.power(input,np.arange(self.degree))*self.weight)
            return output
        else:
            raise ValueError("Unknown type:%s"%type(input))