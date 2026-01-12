#!/usr/bin/env python
"""
model.py
--------
Physics Informed Neural Network for solving Poisson equation
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from problem import Problem
from options import Options
from utils import grad

#define the activate function swish
class Swish(nn.Module):
	def __init__(self, inplace=True):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x * torch.sigmoid(x)

class Net(nn.Module):
    """
    Basic Network for PINNs
    """

    def __init__(self, args):
        """
        Initialization for Net
        """
        super().__init__()
        self.args = args
        self.layers = args.layers  # layers of network
        self.scale = args.scale
        self.device = args.device
        self.problem = args.problem
        self.fcs = []
        self.params = []

        for i in range(len(self.layers) - 2):
            fc = nn.Linear(self.layers[i], self.layers[i+1])
            setattr(self, f'fc{i+1}', fc)
            self._init_weights(fc)
            self.fcs.append(fc)

            param = nn.Parameter(torch.randn(self.layers[i+1]))
            setattr(self, f'param{i+1}', param)
            self.params.append(param)

        fc = nn.Linear(self.layers[-2], self.layers[-1])
        setattr(self, f'fc{len(self.layers)-1}', fc)
        self._init_weights(fc)
        self.fcs.append(fc)

    def _init_weights(self, layer):
        init.xavier_normal_(layer.weight)
        init.constant_(layer.bias, 0.01)

    def forward(self, xyt_kesi):

        xmin, xmax, ymin, ymax, tmin, tmax = self.problem.domain

        # Normalized
        lb = np.ones(23)*(-1)
        lb[0], lb[1], lb[2] = xmin, ymin, tmin
        ub = np.ones(23)
        ub[0], ub[1], ub[2] = xmax, ymax, tmax
        lb = torch.from_numpy(lb).float().to(self.device)
        ub = torch.from_numpy(ub).float().to(self.device)       
        X = 2.0 * (xyt_kesi - lb) / (ub - lb) -1
        X[:, 3:] = xyt_kesi[:, 3:]

        swish = Swish()

        # Y = torch.tanh(self.fcs[0](X))
        Y = swish(self.fcs[0](X)*self.scale)

        for i in range(1, len(self.fcs)-1):
            # Y = torch.tanh(self.fcs[i](Y))
            Y = swish(self.fcs[i](Y)*self.scale)
            #res = self.fcs[i](X)
            #res = torch.mul(self.params[i], res) * self.scale
            #res = torch.sin(res)
            #X = X + res

        u = self.fcs[-1](Y)

        hstar = 80

        y = xyt_kesi[:, [1]]
        t = xyt_kesi[:, [2]]
        dist = ((t-tmin)*(ymax-y)*(y-ymin))/((tmax-tmin)*(ymax-ymin)**2)
        h = hstar + dist * u

        return h


class Net_Neumann(nn.Module):
    """Network for Neumann boundary"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, X_bdy2):

        X_bdy2.requires_grad_(True)

        h = self.net(X_bdy2)
        hx_bdy2 = grad(h, X_bdy2)[:, [0]]

        X_bdy2.detach_()

        return hx_bdy2


class Net_PDE(nn.Module):
    """Network for PDE"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.problem = net.problem
        self.device = net.device

    def forward(self, X):
        """
        Parameters:
        -----------
        X: (n, 23) tensor
            interior points
        """
        X.requires_grad_(True)

        h = self.net(X)
        dh = grad(h, X)
        h_x, h_y, h_t = dh[:, [0]], dh[:, [1]], dh[:, [2]]
        h_xx = grad(h_x, X)[:, [0]]
        h_yy = grad(h_y, X)[:, [1]]
        h_t = dh[:, [2]]

        X.detach_()

        K, K_x, K_y = self.problem.k(X.cpu().numpy())
        #f = self.problem.f(xyt.cpu().numpy())
        #K = K.to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        K_x = K_x.to(self.device)
        K_y = K_y.to(self.device)

        f = self.problem.f(X[:, :2].cpu().numpy())
        f = torch.from_numpy(f).float().to(self.device)
        mu = self.problem.mu

        return mu*h_t-K*(h*(h_xx+h_yy)+h_x*h_x+h_y*h_y)-h*(K_x*h_x+K_y*h_y)-f


class PINN(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.net_Neumann = Net_Neumann(net)
        self.net_PDE = Net_PDE(net)

    def forward(self, X, X_bdy2):

        res = self.net_PDE(X)
        hx_bdy2 = self.net_Neumann(X_bdy2)

        return res, hx_bdy2


if __name__ == '__main__':
    args = Options().parse()
    args.problem = Problem(sigma=args.sigma)

    net = Net(args)
    print(net)

    # net_Neumann = Net_Neumann(net)
    # net_pde = Net_PDE(net, problem)
    # pinn = PINN(net, problem)
    # params = list(net.parameters())

    # for name, value in net.named_parameters():
    #     print(name)
    # print(net.param1.shape)
