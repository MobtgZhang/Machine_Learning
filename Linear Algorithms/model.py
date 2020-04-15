import numpy as np
def cholesky_LLT(A):
    if A.ndim!=2 :
        raise ValueError("size of matrix doesn't match!")
    m,n = A.shape
    if m!=n:
        raise ValueError("size of matrix doesn't match!")
    # calculate the matrix L
    L = np.zeros(A.shape,dtype = np.float)
    n = A.shape[0]
    for i in range(0,n):
        for j in range(0,n):
            if i < j+1:
                if j>0:
                    L[j,j] = np.sqrt(A[j,j]-np.sum(L[j,0:j]*L[j,0:j]))
                else:
                    L[j,j] = np.sqrt(A[j,j])
            else:
                if j>0:
                    L[i,j] = (A[i,j] - np.sum(L[i,0:j]*L[j,0:j]))/L[j,j]
                else:
                    L[i,j] = A[i,j]/L[j,j]
    return L
def cholesky_LLT_solve(A,b):
    n,_ = A.shape
    L = cholesky_LLT(A)
    y = np.zeros(n)
    x = y.copy()
    for i in range(0,n):
        if i == 0:
            y[i] = b[i]/L[i,i]
        else:
            y[i] = (b[i] - np.sum(L[i,0:i]*y[0:i]))/L[i,i]
    for i in range(n-1,-1,-1):
        if i == n-1:
            x[i] = y[i]/ L[i,i]
        else:
            x[i] = (y[i] - np.sum(L[i+1:n,i]*x[i+1:n]))/L[i,i]
    return x
def cholesky_LDLT(A):
    if A.ndim != 2:
        raise ValueError("size of matrix doesn't match!")
    m, n = A.shape
    if m != n:
        raise ValueError("size of matrix doesn't match!")
    # calculate the matrix L
    L = np.zeros(A.shape, dtype=np.float)
    T = np.zeros(A.shape,dtype=np.float)
    n = A.shape[0]
    D = np.zeros(n,dtype=np.float)
    for i in range(0,n):
        if i >0:
            for j in range(0,n):
                if i<j+1:
                    pass
                else:
                    if j > 0:
                        T[i, j] = A[i, j] - np.sum(T[i, 0:j] * L[j, 0:j])
                        L[i,j] = T[i,j]/D[j]
                    else:
                        T[i, j] = A[i, j]
                        L[i,j] = T[i,j]/D[j]
            D[i] = A[i,i] - np.sum(T[i,0:i]*L[i,0:i])
        else:
            D[i] = A[i,i]
    return L,D
def cholesky_LDLT_solve(A,b):
    n, _ = A.shape
    L,D = cholesky_LDLT(A)
    y = np.zeros(n)
    x = y.copy()
    for i in range(0,n):
        if i>0:
            y[i] = b[i] - np.sum(L[i,0:i]*y[0:i])
        else:
            y[i] = b[i]
    for i in range(n-1,-1,-1):
        if i == n-1:
            x[i] = y[i]/D[i]
        else:
            x[i] = y[i]/D[i]-np.sum(L[i+1:n,i]*x[i+1:n])
    return x
