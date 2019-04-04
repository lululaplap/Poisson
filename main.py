import numpy as np
from scipy import signal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filter
from mpl_toolkits.mplot3d import Axes3D
from Jacobi import Jacobi as Poisson
fig = plt.figure()
ax = fig.add_subplot(111)#,projection='3d')

def main():
    N = 10
    x, y = np.meshgrid(np.linspace(-1,1,N), np.linspace(-1,1,N))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.1
    g= np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    phi = np.random.uniform(-0.01,0.01,size=(N,N,N))
    rho = np.zeros((N,N,N))
    m = int(N/2)
    rho[m,m,m] = 1

    P = Poisson(N,phi,rho)
    #ani = animation.FuncAnimation(fig, P.update)
    P.sim(10*10**(-6))

    x, y, z = np.meshgrid(np.linspace(0, P.N, P.N),
                  np.linspace(0, P.N, P.N),
                  np.linspace(0, P.N, P.N))

    norm = -1*np.sum(P.E[:])
    normB = np.sum(P.B)
    u = P.E[0]/norm
    v = P.E[1]/norm
    q = P.E[2]/norm
    #ax.quiver(x,y,z,z,v,u, normalize=True)
    plt.quiver(q[P.m,:,:],v[P.m,:,:],angles='xy',scale=None,pivot='tip')
    #plt.quiver(P.B[2][:,P.m,:],P.B[0][:,P.m,:],angles='uv',scale=None)#,pivot='tip')
    #plt.quiver(P.B[0]/normB,P.B[1]/normB,angles='uv',scale=None,pivot='mid')
    # print(np.shape(P.Em))
    # print(P.Em)
    plt.show()
    plt.imshow(P.phi[:,m,:])#/np.sum(P.phi[m,:,:]))
    plt.show()
main()
