import numpy as np
from scipy import signal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filter
from mpl_toolkits.mplot3d import Axes3D
from Jacobi import Jacobi as Poisson
from GS_Poisson import GS
fig = plt.figure()
ax = fig.add_subplot(111)#,projection='3d')


def main():

    N = 20
    phi = np.random.uniform(-0.01,0.01,size=(N,N,N))
    rho = np.zeros((N,N,N))
    m = int(N/2)
    rho[m,m,m] = 1
    P = GS(N,phi,rho)

    print(P.sim(1**(-4),w=1))
    #P.SOR(10**(-4),n=5)
    P.calcFeild()
    P.plot()


main()
