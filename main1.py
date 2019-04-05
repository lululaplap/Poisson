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

def main(args):
    if len(args)!=6:
        print("[N:,int][Type:E,B],[Method: Jacobi,Gauss],[Run:plot,distance,SOR][Threshold:float]")
        exit()
    N = int(args[1])
    phi = np.random.uniform(-0.01,0.01,size=(N,N,N))
    rho = np.zeros((N,N,N))
    m = int(N/2)
    thresh = float(args[5])

    if str(args[2]) == 'E':
        rho[m,m,m] = 1
    elif str(args[2]) == 'B':
        rho[:,m,m] = 1
    if str(args[3]) == 'Jacobi':
        P= Jacobi(N,phi,rho)
    elif str(args[3]) == 'Gauss':
        P = GS(N,phi,rho)
    if str(args[4]) == 'plot':
        P.calcFeild()
        if str(args[2]) == 'E':
            P.plotE()
        elif str(args[2]) == 'B':
            P.plotB()
    elif str(args[4]) == 'distance':
        d = P.dist()
        P.sim(thresh)
        P.calcFeild()
        if str(args[2]) == 'E':
            plt.scatter(np.log(d[:,:,:].reshape(-1)),np.log(P.phi[:,:,:].reshape(-1)))
        elif str(args[2]) == 'B':
            plt.scatter(np.log(d[m,:,:].reshape(-1)),np.log(P.phi[m,:,:].reshape(-1)))

    elif str(args[4]) == 'SOR':
        P.SOR(10**(-5),50,N,phi,rho)

    plt.show()





main(sys.argv)
