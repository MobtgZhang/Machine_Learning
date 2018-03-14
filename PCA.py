import numpy as np
# Every colum as a sample
# The step of the value :
# (1) make the value zero mean
# (2) calculate covariance matrix
# (3) finding eigenvalues, feature matrices
# (4) Retain the main components 
# [e.g the top n features with relatively large retention values]
def zeroMean(dataMat):
    MeanVal = np.mean(dataMat,axis=0)
    newData = dataMat - MeanVal
    return newData,MeanVal
def pca_num(dataMat,n):
    newData,MeanVal = zeroMean(dataMat)
    covMat = np.cov(newData,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    n_eigVect = eigVects[:,n_eigValIndice]
    lowDDataMat = newData*n_eigVect
    reconMat = (lowDDataMat*n_eigVect.T) + MeanVal
    return lowDDataMat,reconMat
def pca_percentage(dataMat,percentage = 0.99):
    def percentage2n(eigVals,percentage):
        sortArr = np.sort(eigVals)
        sortArr = sortArr[-1::-1]
        arraySum = sum(sortArr)
        tmpSum = 0
        num = 0
        for k in sortArr:
            tmpSum += k
            num += 1
            if tmpSum >= arraySum*percentage:
                return num
    newData,MeanVal = zeroMean(dataMat)
    covMat = np.cov(newData,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    n = percentage2n(eigVals,percentage)
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    n_eigVect = eigVects[:,n_eigValIndice]
    lowDDataMat = newData*n_eigVect
    reconMat = (lowDDataMat*n_eigVect.T) + MeanVal
    return lowDDataMat,reconMat
def main():
    from sklearn import datasets
    digits = datasets.load_iris()
    x = digits.data
    y = digits.target
    outA,outB = pca_percentage(x)
    print(outA)
    print(outB)
if __name__ == "__main__":
    main()
