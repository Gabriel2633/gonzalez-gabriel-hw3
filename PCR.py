import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy as sp 
from matplotlib.colors import LogNorm
from scipy.fftpack import fft, fftfreq, fft2, fftshift, ifft

#%% punto 1
a = plt.imread('arbol.png')

#%% transformada de fourier 
b = fft2(a)

#%%
frec=fftfreq(len(b[0].real))

plt.plot(frec,b)
plt.savefig('gonzalezgabriel_FT2D.pdf')
#%%

for i in range(len(b)):
    for j in range(len(b)):
        if(b[i][j]>4000 and b[i][j]<5000):
            b[i][j]=0



plt.plot(frec,b)
plt.show()

plt.imshow(np.abs(b), norm=LogNorm(vmin=5))
plt.colorbar()
plt.savefig('gonzalezgabriel_FT2D_filtrada.pdf')
#%%

c=ifft(b)

for i in range(len(c)):
    for j in range(len(c)):
        if(c[i][j]>4000 and c[i][j]<5000):
            c[i][j]=0

plt.plot(frec,c)
plt.savefig('gonzalezgabriel_Imagen_filtrada.pdf')





