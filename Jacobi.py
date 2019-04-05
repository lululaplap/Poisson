import numpy as np
from scipy import signal
import scipy.ndimage.filters as filter
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Jacobi():
    def __init__(self,N,phi,rho):

        self.phi = phi
        self.rho = rho
        self.N = N
        self.m = int(self.N/2)
        self.E = np.gradient(self.phi)
        #self.phi = np.reshape(self.phi,(self.N,self.N,self.N))
        self.kernal = (1/6.0)*np.array([[[0,0,0], [0,1,0], [0,0,0]] ,[[0,1,0], [1,0,1], [0,1,0]] ,[[0,0,0], [0,1,0], [0,0,0]]])

    def step(self,w):
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
    def calcFeild(self,norm=False):
        self.E = np.gradient(self.phi)

        gi, gj = np.gradient(self.phi[self.m,:,:])
        self.B = -1*np.array([gi,-gj,np.zeros((self.N,self.N))])
        if norm:
            normB= np.sqrt(self.B[0]**2+self.B[1]**2)#np.linalg.norm(self.B)
            normE= np.sqrt(self.E[0]**2+self.E[1]**2+self.E[2]**2)#np.linalg.norm(self.B)
        else:
            normB =1
            normE = 1
        self.B /= normB
        u = self.E[0]/normE
        v = self.E[1]/normE
        q = self.E[2]/normE
        self.E  = [q,v,u]

        np.savetxt("E.csv",self.E,delimiter=",")
        np.savetxt("B.csv",self.B,delimiter=",")
        return [self.E,self.B]

    def sim(self,thresh=0.001,w=1):
        error = np.inf
        i=0
        maxIter = 250
        while error > thresh and i<maxIter:

            old = self.phi
            self.phi = self.step(w)
            error = np.max(np.abs(self.phi-old))
            print(error)
            i+=1
        return(i)

    def dist(self):
        inds = np.indices((self.N,self.N,self.N))
        print(inds.shape)
        x = np.power((inds[0]-self.m),2)
        y = np.power((inds[1]-self.m),2)
        z = np.power((inds[2]-self.m),2)
        dists = np.sqrt(x+y+z)
        return dists

    def plotB(self):
        plt.quiver(self.B[0],self.B[1],angles='xy',scale=None)
        plt.imshow(self.phi[self.m,:,:],cmap='cool')#/np.sum(P.phi[m,:,:]))

    def plotE(self):


        #plt.quiver(q[self.m,2:-3,2:-3],v[self.m,2:-3,2:-3],angles='xy',scale=None,pivot='tip',color='r')
        plt.quiver(self.E[0][self.m,:,:],self.E[1][self.m,:,:],angles='xy',scale=None)
        plt.imshow(self.phi[self.m,:,:],cmap='cool')#/np.sum(P.phi[m,:,:]))
