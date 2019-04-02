import numpy as np
from main import Poisson

class GS(Poisson):
    def __init__(self,N,phi,rho):
        super().__init__(N,phi,rho)
        self.kernal2 = np.array([[[1,1,1], [1,1,1], [1,1,1]] ,[[1,1,1], [1,1,0], [1,0,1]] ,[[1,1,1], [1,0,1], [1,1,1]]])
        self.kernal1 = np.array([[[1,1,1], [1,0,1], [1,1,1]] ,[[1,0,1], [0,1,1], [1,1,1]] ,[[1,1,1], [1,1,1], [1,1,1]]])
        # print(self.kernal1+self.kernal2==self.kernal)
        self.mask1 = np.ones((self.N,self.N,self.N))
        # print(self.mask1[0:3,0:3,0:3])
        # print(np.shape(self.mask1[0:2,0:2,0:2]))
        # print(self.kernal1)
        # print(np.shape(self.kernal1))
        self.mask1[0:3,0:3,0:3]= self.kernal1
        self.mask2 = np.ones((self.N,self.N,self.N))
        self.mask2[0:3,0:3,0:3]= self.kernal2
        self.phi_next = self.phi


    def step(self):
        for i in range(1,self.N-1):

            for j in range(1,self.N-1):

                for k in range(1,self.N-1):
                    ma1 = np.ma.array(self.phi_next,mask=self.mask1)
                    ma2 = np.ma.array(self.phi,mask=self.mask2)
                    #print(ma1+ma2)
                    self.phi_next[i,j,k] = 1/6.0*(np.sum(ma1)+np.sum(ma2)+self.rho[i,j,k])
                    #print(np.sum(ma1+ma2))
                    self.mask1 = np.roll(self.mask1,1,axis=2)
                    self.mask2 = np.roll(self.mask1,1,axis=2)
                self.mask1 = np.roll(self.mask1,1,axis=1)
                self.mask2 = np.roll(self.mask1,1,axis=1)
            self.mask1 = np.roll(self.mask1,1,axis=0)
            self.mask2 = np.roll(self.mask1,1,axis=0)


        self.E = np.gradient(self.phi)
        gi, gj = np.gradient(self.phi[self.m,:,:])
        self.B = np.array([gi,-gj,np.zeros((self.N,self.N))])
