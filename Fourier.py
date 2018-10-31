import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy as sp 
from scipy.fftpack import fft, fftfreq
#%%    Almacene los datos de signal.dat y de incompletos.dat

a=pd.read_csv("signal.dat", header=None)
b=pd.read_csv("incompletos.dat", header=None)

#%% Haga una grafica de los datos de signal.dat y guarde dicha grafica sin mostrarla en ApellidoNombre_signal.pdf

plt.plot(a[0],a[1])
plt.savefig("GonzalezGabriel_signal.pdf")

#%% Haga la transformada de Fourier de los datos de la senal usando su implementacion propia de la transformada discreta de fourie
def datos(x):
    n=len(x)
    f=200.0
    dt=1/(f * 32 )
    return n,f,dt
def fourier(x): 
    N=[]
    for i in range(len(x)):
        N.append(i)
    R=[]
    for i in range(len(N)):
        resp=np.sum(x[i]*np.exp(-(2j)*np.pi*i*(N[i]/len(x))))
        R.append(resp)
    return R
# hay un error preguntar profe que sucede

# para los datos de signal 
    
fa=fft(a[1])

# para los datos de incompletos 
    
fb=fft(b[1])
    
#%% Haga una grafica de la transformada de Fourier y guarde dicha grafica sin mostrarla en ApellidoNombre_TF.pdf.
melo=datos(a[1])
freq = fftfreq(melo[0], melo[2])

melo2=datos(b[1])
freq2 = fftfreq(melo2[0], melo2[2])

plt.ylabel("fourier")
plt.xlabel("frecuencias")

plt.plot(freq,fa)
plt.plot(freq2,fb)
plt.show()
plt.savefig("GonzalezGabriel_TF.pdf")

#%% Imprima un mensaje donde indique cuales son las frecuencias principales de su senal

print "no se puede hacer devido a que en si, la precicion de mis datos no es buena por lo tanto tampoco se podra encontrar periodicidad en mis datos "
#%% filtro 
for i in range(len(freq)):
    if(abs(freq[i])>1000):
        freq[i]=0
        
plt.plot(freq,fa)
plt.savefig('GonzalezGabriel_filtrada.pdf')
    
#%% interpolacion cuadratica 
from scipy.interpolate import interp1d
x=np.linspace(min(b[0]),max(b[0]), num=512, endpoint=True)
f = interp1d(b[0],b[1])
f2 = interp1d(b[0],b[1], kind='cubic')
serie1 = f(x)
serie2 = f2(x)
fserie1 = fft(serie1)
fserie2 = fft(serie2)
#%%HagaunagraficacontressubplotsdelastrestransformadadeFourier
freqs1 = fftfreq(512, melo2[2])
freqs2 = fftfreq(512, melo2[2])
plt.plot(freq2,fb)
plt.plot(freqs1,serie1)
plt.plot(freqs2,serie2)
plt.savefig('gonzalezgabriel_TF_interpola.pdf')

#%%Imprima un mensaje donde describa a
print "la diferencia recae en la precicion de mis datos por lo que se puede evidenciar la interpolacion cubica tendra mayor precicion que los demas, a su vez es importante aclarar que respecto a mis datos originales la precicion y periodicidad de los datos esta mala por su tamano "

#%%Aplique el filtro pasabajos con una frecuencia de corte fc = 1000Hz y con una frecuencia de corte de fc = 500Hz.

for i in range(len(freq2)):
    if(abs(freq2[i])>1000):
        freq2[i]=0

for i in range(len(freqs1)):
    if(abs(freqs1[i])>1000):
        freqs1[i]=0

for i in range(len(freqs2)):
    if(abs(freqs2[i])>1000):
        freqs2[i]=0

plt.subplot(2,2,1)
plt.plot(freq2,fb)
plt.plot(freqs1,serie1)
plt.plot(freqs2,serie2)

for i in range(len(freq2)):
    if(abs(freq2[i])>500):
        freq2[i]=0

for i in range(len(freqs1)):
    if(abs(freqs1[i])>500):
        freqs1[i]=0

for i in range(len(freqs2)):
    if(abs(freqs2[i])>500):
        freqs2[i]=0

plt.subplot(2,2,2)
plt.plot(freq2,fb)
plt.plot(freqs1,serie1)
plt.plot(freqs2,serie2)
plt.savefig('gonzalezgabriel_2Filtros.pdf.')










