import numpy as np
from scipy import signal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filter

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

class Poisson():
    def __init__(self,N):
        self.N = N
        self.phi = np.random.uniform(-0.01,0.01,size=(self.N**3))
        self.rho = np.zeros((self.N,self.N,self.N))
        self.m = int(self.N/2)
        m = self.m
        self.rho[m,m,m] = 1
        print(np.shape(self.phi))

        self.phi = np.reshape(self.phi,(self.N,self.N,self.N))
        self.kernal = (1/6.0)*np.array([[[0,0,0], [0,1,0], [0,0,0]] ,[[0,1,0], [1,0,1], [0,1,0]] ,[[0,0,0], [0,1,0], [0,0,0]]])


        print(np.shape(self.phi))
        print(np.shape(self.kernal))

    def update(self,i):
        # print(np.shape(self.phi))
        # print(np.shape(self.kernal))
        # print(self.phi)
        for i in range(10):
            self.phi = filter.convolve(self.phi,self.kernal, mode='constant',cval=0)+self.rho/6
        ax.clear()
        ax.imshow(self.phi[self.m,:,:])

    def sim(self):
        for i in range(100):
            self.update(i)

def main():
    P = Poisson(100)
    ani = animation.FuncAnimation(fig, P.update)
    plt.show()
    #P.sim()
main()
