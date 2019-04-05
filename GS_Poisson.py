import numpy as np
from Jacobi import Jacobi as Poisson
import matplotlib.pyplot as plt

class GS(Poisson):
    def __init__(self,N,phi,rho):
        super().__init__(N,phi,rho)
        self.phi = np.pad(self.phi,((1,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
        self.rho = np.pad(self.rho,((1,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
        self.phi_next = self.phi

    def calcFeild(self):
        self.phi = self.phi[1:-1,1:-1,1:-1]
        return super().calcFeild()


    def step(self,w=1):

        self.phi_next = np.copy(self.phi)
        for i in range(1,self.N+1):
            for j in range(1,self.N+1):
                for k in range(1,self.N+1):
                    old = self.phi[i,j,k]
                    GS =    1/6.0*(self.phi[(i+1),j,k]+self.phi_next[(i-1),j,k] \
                                 +self.phi[i,(j+1),k]+self.phi_next[i,(j-1),k] \
                                 +self.phi[i,j,(k+1)]+self.phi_next[i,j,(k-1)] \
                                 +self.rho[i,j,k])
                    self.phi_next[i,j,k] = (1-w)*old + w*GS
        return(self.phi_next)#thank u, phi next

    @classmethod
    def SOR(cls,thresh,n,N,phi,rho):
        w = np.linspace(1,2,n)
        x = np.zeros(n)
        for i in range(0,n):
            P = cls(N,phi,rho)
            x[i] = P.sim(thresh,w=w[i])
            print("--------------{}/{}-------------".format(i+1,n))
        plt.plot(w,x)
        plt.title("SOR")
        plt.xlabel("w")
        plt.ylabel("Iterations")
        plt.show()
