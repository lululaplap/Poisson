import numpy as np
from scipy import signal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filter
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111)#,projection='3d')

class Poisson():
    def __init__(self,N,phi,rho):

        self.phi = phi
        self.rho = rho
        self.N = N
        self.m = int(self.N/2)
        self.E = np.gradient(self.phi)
        #self.phi = np.reshape(self.phi,(self.N,self.N,self.N))
        self.kernal = (1/6.0)*np.array([[[0,0,0], [0,1,0], [0,0,0]] ,[[0,1,0], [1,0,1], [0,1,0]] ,[[0,0,0], [0,1,0], [0,0,0]]])

    def step(self):
        self.phi = filter.convolve(self.phi,self.kernal, mode='constant',cval=0)+self.rho/6.0
        self.E = np.gradient(self.phi)
        gi, gj = np.gradient(self.phi[self.m,:,:])
        self.B = np.array([gi,-gj,np.zeros((self.N,self.N))])

    def update(self,i):
        for j in range(100):
            self.step()
        ax.clear()
        #ax.imshow(self.phi[self.m,:,:])
        x, y, z = np.meshgrid(np.linspace(0, self.N, self.N),
                      np.linspace(0, self.N, self.N),
                      np.linspace(0, self.N, self.N))
        print(np.shape(self.E[0][self.m,:,:]))
        ax.quiver(x,y,z,self.E[1],self.E[0],self.E[2])

    def sim(self,n):
        for i in range(n):
            self.step()

def main():
    N = 100
    phi = np.random.uniform(-0.01,0.01,size=(N,N,N))
    rho = np.zeros((N,N,N))
    m = int(N/2)
    rho[m,m,m] = 1

    P = Poisson(N,phi,rho)
    #ani = animation.FuncAnimation(fig, P.update)
    P.sim(100)

    x, y, z = np.meshgrid(np.linspace(0, P.N, P.N),
                  np.linspace(0, P.N, P.N),
                  np.linspace(0, P.N, P.N))

    norm = -1*np.sum(P.E[:])
    normB = np.sum(P.B)
    u = P.E[0]/norm
    v = P.E[1]/norm
    q = P.E[2]/norm
    #ax.quiver(x,y,z,z,v,u, normalize=True)
    #plt.quiver(q[P.m,:,:],v[P.m,:,:],angles='xy',scale=None,pivot='tip')
    #plt.quiver(P.B[2][:,P.m,:],P.B[0][:,P.m,:],angles='uv',scale=None)#,pivot='tip')
    plt.quiver(P.B[0]/normB,P.B[1]/normB,angles='uv',scale=400*100000,pivot='mid')
    # print(np.shape(P.Em))
    # print(P.Em)
    plt.show()
    plt.imshow(P.phi[:,m,:])#/np.sum(P.phi[m,:,:]))
    plt.show()
main()
