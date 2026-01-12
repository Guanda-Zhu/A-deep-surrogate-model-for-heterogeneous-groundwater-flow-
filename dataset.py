#!/usr/bin/env python
from options import Options
from utils import tile, tile_kesi
from pyDOE import lhs
from sampler import Sampler
from problem import Problem
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class Trainset(Dataset):
    """Generate DataSet for Training

    Params
    ------
    problem  (class Problem)
        used for determine some properties of the problem

    fname (class str) mesh fname
        used for generate interior and boundary points (nodes) """

    def __init__(self, problem, nt=50, n_kesi=180, fname=None):
        """
        Obtain the interior points and boundary points from specified file using Dataprocessing

        Members:
        -------
        self.xyt:      (n, 2) ndarray

        self.xyt_bdy1: (n_bdy1, 2) ndarray
        self.u_bdy1:   (n_bdy1, 2) ndarray

        self.xyt_bdy2: (n_bdy2, 2) ndarray
        self.u_bdy2:   (n_bdy2, 2) ndarray
        """
        self.problem = problem
        self.nt, self.n_kesi = nt, n_kesi
        self.fname = fname

        sampler = Sampler(problem, nt=nt, n_kesi=n_kesi, fname=fname)

        self.xyt, self.xyt_bdy2, self.kesi = sampler()

        self.u_bdy2 = problem.bc(self.xyt_bdy2, mode=2)

    def __getitem__(self, index):
        # 生成一个随机整数
        rand_int = np.random.randint(low=0, high=1000)
        X = tile_kesi(self.xyt, index)
        # X = tile_kesi(self.xyt, self.kesi[index].reshape(1,-1))
        res = np.zeros_like(X[:, [0]])
        X_bdy2 = tile_kesi(self.xyt_bdy2, index)
        # X_bdy2 = tile_kesi(self.xyt_bdy2, self.kesi[index].reshape(1,-1))
        u_bdy2 = tile(self.u_bdy2, self.kesi[index].reshape(1,-1))[:, [0]]

        return X, res, X_bdy2, u_bdy2

    def __len__(self):
        return self.n_kesi

    def __repr__(self):
        res = f'''************** Trainset *****************\n''' + \
            f'''Spatial INFO: file={self.fname}\n''' + \
            f'''Temporal INFO: nt={self.nt}\n''' + \
            f'''xyt      = {self.xyt.shape}\n''' + \
            f'''xyt_bdy2 = {self.xyt_bdy2.shape}\n''' + \
            f'''kesi        = {self.kesi.shape}\n''' + \
            f'''****************************************'''
        return res

    def __call__(self, mode=None):
        if mode == 0:
            return self.xy0, self.u0
        elif mode == 1:
            return self.xyt_bdy1, self.u_bdy1
        elif mode == 2:
            return self.xyt_bdy2, self.u_bdy2
        elif mode is None:
            return self.xyt


class Validset():
    """
    Generate DataSet for Training

    Params:
    ------
    problem (class Problem)
        used for determine the solution domain

    nx, ny  (class int)
        the number of partitions in x and y direction

    t: (int, float, list, tuple, ndarray)
       int: number of temporal points to get uniform points
       (float, list, tuple, ndarray): given temporal points

    kesi_seed: (int, float, list, tuple, ndarray)
       int: random seed for generating hydrualic conductivity field
       (float, list, tuple, ndarray): given random seed for hydrualic conductivity field

    """

    def __init__(self, problem, nx, ny, t, kesi_seed):
        """Obtain the sample points for validation

        Params
        ======
        size:
        """
        self.problem = problem
        self.nx = nx
        self.ny = ny
        self.t = t
        self.kesi_seed = kesi_seed

    def spatial(self, grid=False):
        """generation of spatial sampling points"""
        x_min, x_max, y_min, y_max, _, _ = self.problem.domain

        x = np.linspace(x_min, x_max, self.nx)
        y = np.linspace(y_min, y_max, self.ny)
        grid_x, grid_y = np.meshgrid(x, y)
        x_, y_ = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
        xy = np.c_[x_, y_]

        if grid:
            return grid_x, grid_y
        return xy

    def temporal(self):
        """generation of temporal sampling points"""
        if isinstance(self.t, int):
            _, _, _, _, t_min, t_max = self.problem.domain
            t = np.linspace(t_min, t_max, self.t+1)[1:]
        elif isinstance(self.t, float):
            t = np.array([self.t])
        elif isinstance(self.t, (list, tuple)):
            t = np.array(self.t)
        elif isinstance(self.t, np.ndarray):
            t = self.t

        return np.expand_dims(t, axis=1)

    def conductivity(self):
        """generation of random variables"""

        np.random.seed(self.kesi_seed)
        kesi_valid = np.random.randn(1,20)

        return kesi_valid

    def spatial_temporal(self):
        """generation of spatial-temporal sampling points"""

        xy = self.spatial()
        t = self.temporal()

        xt = tile(xy[:, [0]], t)
        yt = tile(xy[:, [1]], t)
        return np.hstack((xt[:, [0]], yt[:, [0]], yt[:, [1]]))

    def spatial_temporal_conductivity(self):
        xyt = self.spatial_temporal()
        kesi = self.conductivity()
        xyt_kesi = tile(xyt, kesi)
        return xyt_kesi

    def __call__(self):
        xyt_kesi = self.spatial_temporal_conductivity()
        return torch.from_numpy(xyt_kesi).float()

    def __repr__(self):
        res = f'''\n*********** Validset *************\n''' + \
            f'''Spatial INFO: nx={self.nx}, ny={self.ny}\n''' + \
            f'''Temporal INFO: t = {self.temporal()}\n''' + \
            f'''Conductivity INFO: seed_k = {self.kesi_seed}\n''' + \
            f'''**********************************\n'''
        return res


if __name__ == '__main__':
    args = Options().parse()
    problem = Problem()

    # Trainset
    trainset = Trainset(problem, fname='./data/well.mat')

    dataloader = DataLoader(trainset, batch_size=10, shuffle=True)
    for data in dataloader:
        X, res, X_bdy2, u_bdy2 = data
        X = X.view(-1, 23).float()
        res = res.view(-1, 1).float()
        X_bdy2 = X_bdy2.view(-1, 23).float()
        u_bdy2 = u_bdy2.view(-1, 1).float()
        print(X.shape, res.shape, X_bdy2.shape, u_bdy2.shape)
        print(X.dtype, res.dtype, X_bdy2.dtype, u_bdy2.dtype)

    # Validset
    validset = Validset(problem, 50, 50, [0.5, 0.75], 38)
    d = validset()
    print(validset, validset())
