import numpy as np
def GaussianSeq(Mat,Bais):
    if (Mat.shape[0]!=Mat.shape[1]) or (Mat.shape[0]!=Bais.shape[0]):
        raise ValueError("size of matrix doesn't match!")
    A = Mat.copy()
    b = Bais.copy()
    n = A.shape[0]
    # 消元过程
    for k in range(n):
        for i in range(k+1,n):
            l_ik = A[i,k]/A[k,k]
            for j in range(k,n):
                A[i, j] = A[i, j] - l_ik * A[k, j]
            b[i] = b[i] - l_ik*b[k]
    # 回代过程
    x = np.zeros((n,))
    x[n-1] = b[n-1]/A[n-1,n-1]
    for k in range(n-2,-1,-1):
        x[k] = (b[k]-np.sum(A[k,k+1:]*x[k+1:]))/A[k,k]
    return x
def GaussianMain(Mat,Bais):
    if (Mat.shape[0]!=Mat.shape[1]) or (Mat.shape[0]!=Bais.shape[0]):
        raise ValueError("size of matrix doesn't match!")
    A = Mat.copy()
    b = Bais.copy()
    n = A.shape[0]
    # 消元过程
    for k in range(n ):
        # 选定主元素
        # r = np.where(np.abs(A[:,k]) == np.max(np.abs(A[k:,k])))[0][0]
        r = k
        for j in range(k,n):
            if np.abs(A[j,k])>np.abs(A[r,k]):
                r = j
        # 交换行
        A[[r, k], :] = A[[k, r], :]
        b[r],b[k] = b[k],b[r]
        if np.abs(A[k,k])< 1e-150:
            raise np.linalg.LinAlgError
        for i in range(k+1,n):
            l_ik = A[i,k]/A[k,k]
            for j in range(k,n):
                A[i, j] = A[i, j] - l_ik * A[k, j]
            b[i] = b[i] - l_ik*b[k]

    # 回代过程
    x = np.zeros((n,))
    x[n-1] = b[n-1]/A[n-1,n-1]
    for k in range(n-2,-1,-1):
        x[k] = (b[k]-np.sum(A[k,k+1:]*x[k+1:]))/A[k,k]
    return x
def Gauss_Jordan(Mat,Bais):
    if (Mat.shape[0] != Mat.shape[1]) or (Mat.shape[0] != Bais.shape[0]):
        raise ValueError("size of matrix doesn't match!")
    A = Mat.copy()
    b = Bais.copy()
    n = A.shape[0]
    # 消元过程
    for k in range(n):
        # 选定主元素
        r = k
        for j in range(k, n):
            if np.abs(A[j, k]) > np.abs(A[r, k]):
                r = j
        # 交换行
        A[[r, k], :] = A[[k, r], :]
        b[r], b[k] = b[k], b[r]
        if np.abs(A[k, k]) < 1e-150:
            raise np.linalg.LinAlgError

        tmp = A[k, k]
        for j in range(k,n):
            A[k,j] = A[k,j]/tmp
        b[k] = b[k]/tmp
        for i in range(n):
            if i!=k:
                tmp = A[i,k]
                for j in range(k,n):
                    A[i,j] = A[i,j] - tmp *A[k,j]
                b[i] = b[i] - tmp*b[k]
    return b
def Doolittle_LU(A):
    if A.shape[0]!=A.shape[1]:
        raise ValueError("size of matrix doesn't match!")
    L_U_mat = np.zeros(A.shape)
    n = A.shape[0]
    for k in range(n):
        for j in range(k,n):
            if k==0:
                L_U_mat[k,j] = A[k,j]
            else:
                L_U_mat[k,j] = A[k,j] - np.sum(L_U_mat[k,:k]*L_U_mat[:k,j])
        for i in range(k+1,n):
            if k==0:
                L_U_mat[i,k] = A[i,k]/L_U_mat[k,k]
            else:
                L_U_mat[i,k] = (A[i,k] -np.sum(L_U_mat[i,:k]*L_U_mat[:k,k]))/L_U_mat[k,k]
    #L = np.tril(L_U_mat,k=-1)
    #U = np.triu(L_U_mat,k=0)
    #L = L + np.diag([1.0]*n)
    return L_U_mat
def Doolittle_solve(A,b):
    if A.shape[0]!=b.shape[0]:
        raise ValueError("size of matrix doesn't match!")
    L_U_mat = Doolittle_LU(A)
    n = L_U_mat.shape[0]
    y = np.zeros((n,))
    for k in range(n):
        if k==0:
            y[k] = b[k]
        else:
            y[k] = b[k] - np.sum(L_U_mat[k,:k]*y[:k])
    x = np.zeros((n,))
    for k in range(n-1,-1,-1):
        if k == n-1:
            x[k] = y[k]/L_U_mat[k,k]
        else:
            x[k] = (y[k] - np.sum(L_U_mat[k,k+1:]*x[k+1:]))/L_U_mat[k,k]
    return x
if __name__ == '__main__':
    n = 5
    A = np.random.rand(n,n)
    b = np.random.rand(n,)
    print(np.linalg.solve(A,b))
    print(GaussianSeq(A,b))
    print(GaussianMain(A,b))
    print(Gauss_Jordan(A,b))
    print(Doolittle_solve(A,b))
