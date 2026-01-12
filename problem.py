#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from kle import KLE


class Problem(object):
    def __init__(self,
                 sigma=25.0,
                 domain=(-500, 500, -500, 500, 0, 10),
                 xi=(0.0, 0.0)):
        self.sigma = sigma
        self.domain = domain
        self.xi = xi
        self.mu = 0.05

    def __repr__(self):
        return f'well problem with point source'

    def f(self, x):
        f = - 2000 * np.exp(-((x[:, [0]] - self.xi[0])**2 +
                                   (x[:, [1]] - self.xi[1])**2) / (2*self.sigma**2)) / (2*np.pi*self.sigma**2)
        return f

    def bc(self, x, mode=0):
        """boundary/initial condition"""

        if mode == 0:  # initial condition
            u0 = 80 * np.ones_like(x[:, [0]])
            return u0

        elif mode == 1:  # Dirichlet boundary condition
            u_bdy1 = 80 * np.ones_like(x[:, [0]])
            return u_bdy1

        elif mode == 2:  # Neumann boundary condition
            u_bdy2 = 0.0 * np.ones_like(x[:, [0]])
            return u_bdy2

    def k(self, xyt_kesi):
        mean_logk = 2.5 #均值
        var1 = 1.0 #方差
        L_x = 1020 #区域长度
        L_y = 1020
        eta=408   #相关长度

        nx = 51
        ny = 51
        delx = 20
        dely = 20
        weight=0.8 #KL展开需要保存的扰动量，80%
        n_eigen=50

        kle = KLE(L_x,L_y,eta,eta,var1,mean_logk,n_eigen,weight,nx,ny,delx,dely)
        kk,logk,k_x,k_y = kle.gen_field_tensor2(xyt_kesi)
        #kk.reshape(1, -1)
        return kk, k_x, k_y

if __name__ == '__main__':
    from sampler import Sampler
    from dataset import Testset

    problem = Problem(sigma=1e-2)
    sampler = Sampler(problem,  'well.mat')
    xyt, xy0, xyt_bdy1, xyt_bdy2 = sampler()

    f = problem.f(xyt)
    print(f.shape)

    g = problem.bc(xy0, mode='initial')
    print(g.shape)
