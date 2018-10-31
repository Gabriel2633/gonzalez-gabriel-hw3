import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

a=pd.read_csv("WDBC.dat.txt", header=None)
del a[1]
del a[0]






def covarianza(x,y):
    meanX= np.mean(x)
    meanY= np.mean(y)
    N = len(x)
    cov = 0  
    for i in range(N):
        cov+=((x[i]-meanX)*(y[i]-meanY))/(N-1)
    return cov


Mcov1 = np.zeros((len(a),len(a)))
for fila in range (2,a.shape[1]):
    for col in range (2,a.shape[1]):
        if(col!=1):
            Mcov1[fila-1, col-1] = covarianza(a[fila],a[col]) 
eig_valores1,eig_vectores1= np.linalg.eig(Mcov1)
pPunto1 = np.dot(Mcov1, eig_vectores1)    

    


Mcovarianza=np.cov(a)







