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

    N = 50
    phi = np.random.uniform(-0.01,0.01,size=(N,N,N))
    rho = np.zeros((N,N,N))
    m = int(N/2)
    rho[m,m,m] = 1
    print(np.shape(rho))
    #P = Poisson(N,phi,rho)
    P = GS(N,phi,rho)
    #ani = animation.FuncAnimation(fig, P.update)
    P.sim(10*10**(-5))
    k = 3
    x, y, z = np.meshgrid(np.linspace(0, P.N, P.N),
                  np.linspace(0, P.N, P.N),
                  np.linspace(0, P.N, P.N))

    norm = -1*np.sum(P.E[:])
    normB = 1#np.sum(P.B)
    u = P.E[0]/norm
    v = P.E[1]/norm
    q = P.E[2]/norm
    #ax.quiver(x,y,z,z,v,u, normalize=True)
    plt.quiver(q[P.m,k-1:N-k,k-1:N-k],v[P.m,k-1:N-k,k-1:N-k],angles='xy',scale=None,pivot='tip',color='r')
    #plt.quiver(P.B[2][:,P.m,:],P.B[0][:,P.m,:],angles='uv',scale=None)#,pivot='tip')
    #plt.quiver(P.B[0]/normB,P.B[1]/normB,angles='uv',scale=400*100000,pivot='mid',color='r')
    # print(np.shape(P.Em))
    # print(P.Em)

    plt.imshow(P.phi[m,k-1:N-k,k-1:N-k])#/np.sum(P.phi[m,:,:]))
    plt.show()

main()
