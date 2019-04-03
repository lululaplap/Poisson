import numpy as np
from Jacobi import Jacobi as Poisson

class GS(Poisson):
    def __init__(self,N,phi,rho):
        super().__init__(N,phi,rho)
        self.phi = np.pad(self.phi,((1 ,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
        self.rho = np.pad(self.rho,((1,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
        self.phi_next = self.phi



    def step(self):
        self.phi_next = np.copy(self.phi)
        self.phi_old = np.copy(self.phi)
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    self.phi_next[i,j,k] = 1/6.0*(self.phi[(i+1),j,k]+self.phi_next[(i-1),j,k] \
                                        +self.phi[i,(j+1),k]+self.phi_next[i,(j-1),k] \
                                        +self.phi[i,j,(k+1)]+self.phi_next[i,j,(k-1)] + self.rho[i,j,k])
        self.phi = self.phi_next

        #self.phi = self.phi_next
        self.E = np.gradient(self.phi)
        gi, gj = np.gradient(self.phi[self.m,:,:])
        self.B = np.array([gi,-gj,np.zeros((self.N,self.N))])
        return(self.phi_next)#thank u, phi next
