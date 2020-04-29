import theano.tensor as T
import theano
import numpy as np
def Gauss(x,mu,gamma):
    return T.exp(-T.square(x-mu)/(2*gamma*gamma))
def Sigmoid(x,mu,gamma):
    return T.nnet.sigmoid(T.square(x-mu)/(gamma*gamma))
def Inverse(x,mu,gamma):
    return 1.0/T.sqrt(T.square(x-mu)+T.square(gamma))
