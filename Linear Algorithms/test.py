import numpy as np
import scipy as sp
import scipy.linalg
from numpy.linalg import solve
import matplotlib.pyplot as plt
from model import cholesky_LDLT_solve,cholesky_LLT_solve
from polynomial import Polynomial
def rand_create(N):
    D = np.diag(np.random.rand(N))
    M = sp.linalg.orth(np.random.rand(N,N))
    out = np.dot(np.dot(M.T,D),M)
    return out
def create_data():
    length = 500
    tip = 8000
    # make up some data
    x_data = np.linspace(-10, 10, length)
    y_data = 10 * np.power(x_data, 3) - 6 * x_data + 8
    noise = tip*np.random.rand(length)
    y_data = y_data +noise
    return x_data,y_data
def test_cholesky():
    N = 3
    A = rand_create(N)
    b = np.random.rand(N)
    x1 = cholesky_LLT_solve(A,b)
    print(x1)
    x2 = cholesky_LDLT_solve(A,b)
    print(x2)
    print(solve(A,b))
def test_Polynomial():
    x_data, y_data = create_data()
    model = Polynomial(6, 0.1)
    model.fit(x_data, y_data)
    input = np.linspace(-10, 10)
    output = model.regress(input)
    plt.scatter(x_data, y_data)
    plt.savefig("pic0.png")
    plt.plot(input, output, c="r")
    plt.savefig("pic1.png")
    plt.show()
if __name__ == '__main__':
    test_cholesky()
    test_Polynomial()
