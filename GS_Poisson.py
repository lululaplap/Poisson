import numpy as np
from Jacobi import Jacobi as Poisson

class GS(Poisson):
    def __init__(self,N,phi,rho):
        super().__init__(N,phi,rho)
        self.phi = np.pad(self.phi,((1,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
        self.rho = np.pad(self.rho,((1,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
        self.phi_next = self.phi

    def calcFeild(self):
        print(self.phi.shape)
        self.phi = self.phi[1:-1,1:-1,1:-1]
        print(self.phi.shape)
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
