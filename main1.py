import numpy as np
from scipy import signal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filter
from mpl_toolkits.mplot3d import Axes3D
from Jacobi import Jacobi
from GS_Poisson import GS
fig = plt.figure()
ax = fig.add_subplot(111)#,projection='3d')
import sys

def main():
    N = 30
    phi = np.random.uniform(-0.01,0.01,size=(N,N,N))
    rho = np.zeros((N,N,N))
    m = int(N/2)
    rho[m,m,m] = 1
    PJ= Jacobi(N,phi,rho)
    PGS = GS(N,phi,rho)
    #print(P.sim(0.0001))#,w=1.75))
    d = P.dist()
    P.sim(0.001)
    #plt.scatter(np.log(d[:,:,:].reshape(-1)),np.log(P.phi[:,:,:].reshape(-1)))
    plt.show()
    #GS.SOR(10**(-5),50,N,phi,rho)
    P.calcFeild()
    P.plot()



main()
