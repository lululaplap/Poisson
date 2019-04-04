import numpy as np
from scipy import signal
import scipy.ndimage.filters as filter
import matplotlib.pyplot as plt

class Jacobi():
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
        return self.phi

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
    def calcFeild(self):
        self.E = np.gradient(self.phi)
        gi, gj = np.gradient(self.phi[self.m,:,:])
        self.B = np.array([gi,-gj,np.zeros((self.N,self.N))])
        return [self.E,self.B]

    def sim(self,thresh,w=1):
        error = np.inf
        i=0
        while error > thresh:
            old = self.phi
            self.phi = self.step(w)
            error = np.mean(np.abs(self.phi-old))
            #error = np.mean(np.abs(self.step(w)))
            #self.phi = self.phi_next
            print(error)
            i+=1
        return(i)
    def SOR(self,thresh,n=20):
        w = np.linspace(1,2,n)
        x = np.zeros(n)
        for i in range(0,n):
            x[i] = self.sim(thresh,w=w[i])
            print(w[i])
        plt.plot(w,x)
        plt.show()
    def plot(self):
        norm = -1*np.sum(self.E[:])
        normB = 1#np.sum(P.B)
        u = self.E[0]/norm
        v = self.E[1]/norm
        q = self.E[2]/norm
        #ax.quiver(x,y,z,z,v,u, normalize=True)
        # plt.quiver(q[self.m,2:-3,2:-3],v[self.m,2:-3,2:-3],angles='xy',scale=None,pivot='tip',color='r')
        #plt.quiver(P.B[2][:,P.m,:],P.B[0][:,P.m,:],angles='uv',scale=None)#,pivot='tip')
        #plt.quiver(P.B[0]/normB,P.B[1]/normB,angles='uv',scale=400*100000,pivot='mid',color='r')
        # print(np.shape(P.Em))
        # print(P.Em)

        #plt.imshow(self.phi[self.m,2:-3,2:-3])#/np.sum(P.phi[m,:,:]))
        plt.quiver(q[self.m,:,:],v[self.m,:,:],angles='xy',scale=None,pivot='tip',color='r')
        plt.imshow(self.phi[self.m,:,:])#/np.sum(P.phi[m,:,:]))
        plt.show()
