#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from options import Options
import os
import scipy.io
import math
from pyDOE import lhs
from utils import tile


class Sampler(object):
    def __init__(self, problem, nt=50, n_kesi=180, fname=None):
        self.problem = problem
        self.sigma = problem.sigma
        self.nt = nt
        self.n_kesi = n_kesi
        self.fname = fname

    def spatial(self):
        """Genration of spatial sampling points"""
        if self.fname is None:
            raise ValueError('a spatial sampling file should be provided!')

        xmin, xmax, ymin, ymax, _, _ = self.problem.domain

        data = scipy.io.loadmat(self.fname)
        xy_bdy = data['bdy2']  # boundary data
        xy = data['xy2']  # interior data

        # boundary of second kind (neumann)
        pos_bdy2 = np.where((xy_bdy[:, 0] == xmin) | (xy_bdy[:, 0] == xmax))
        xy_bdy2 = xy_bdy[pos_bdy2]
        pos_more = np.where((xy_bdy2[:, 1] != ymin) & (xy_bdy2[:, 1] != ymax))
        xy_bdy2 = xy_bdy2[pos_more]

        return xy, xy_bdy2

    def temporal(self):
        """Generation of temporal sampling points"""
        _, _, _, _, tmin, tmax = self.problem.domain
        t = np.linspace(tmin, tmax, self.nt)
        return np.expand_dims(t, axis=1)

    def conductivity(self):

        # np.random.seed(5)
        kesi = np.random.randn(self.n_kesi,20)

        return kesi

    def spatial_temporal(self, xy, t):
        """Generation of spatial_temporal sampling points"""
        xt = tile(xy[:, [0]], t)
        yt = tile(xy[:, [1]], t)
        return np.hstack((xt[:, [0]], yt[:, [0]], yt[:, [1]]))

    def spatial_temporal_conductivity(self, xyt, kesi):
        xyt_kesi = tile(xyt,kesi)
        return xyt_kesi

    def __call__(self):

        xy, xy_bdy2 = self.spatial()
        t = self.temporal()
        kesi = self.conductivity()

        xyt = self.spatial_temporal(xy, t[1:])
        xyt_bdy2 = self.spatial_temporal(xy_bdy2, t[1:, [0]])

        return xyt, xyt_bdy2, kesi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    args = Options().parse()
    from problem import Problem

    problem = Problem()
    sampler = Sampler(problem, fname='./data/well.mat')
    xy, xy_bdy2 = sampler.spatial()
    t = sampler.temporal()
    kesi = sampler.conductivity()
    print(xy.shape, xy_bdy2.shape, t.shape, kesi.shape)

    xyt = sampler.spatial_temporal(xy, t)
    xyt_kesi = sampler.spatial_temporal_conductivity(xyt, kesi)
    print(xyt.shape)
    print(xyt_kesi.shape)
