#!/usr/bin/env python
from options import Options
from utils2 import save_checkpoints, show_surface, show_contours_2x2, show_contours_2x2_2, plot_prediction, mae, mse, rrmse, nse, tile
from model import Net, Net_Neumann, Net_PDE, PINN
from dataset import Trainset, Validset
from sampler import Sampler
from problem import Problem
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import time
import os
import argparse
import shutil


class Tester():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.cuda_index = args.cuda_index
        self.problem = Problem(sigma=args.sigma)

        # Model name
        name = f"{args.hidden_layers}x{args.hidden_neurons}"
        self.model_name = f"{name}_nt:{args.nt}_n_kesi:{args.n_kesi}_bs:{args.batch_size}"

        # Networks
        self.net = Net(self.args)
        self.net_neumann = Net_Neumann(self.net)
        self.net_pde = Net_PDE(self.net)
        self.pinn = PINN(self.net)

        ########################
        # Transfer them to GPU
        ########################
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        # Criterion
        self.criterion = nn.MSELoss()

        # Resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'Resuming training, loading {args.resume} ...')
                self.pinn.load_state_dict(
                    torch.load(args.resume)['state_dict'])
            else:
                print('input resume error', args.resume)

    def test(self, seed_k):
    
        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        name = f"{args.hidden_layers}x{args.hidden_neurons}"
        outpath = os.path.join('.','figures',name, f'seed={seed_k}')
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Validset
        t = [2.5, 5, 7.5, 10]
        # q = [capacity / 10000]
        testset = Validset(self.problem, 100, 100, t, seed_k)
        #testset = Validset(self.problem, 100, 100, t, q)

        X = testset()
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            X = X.to(self.device)

        res = self.net_pde(X)
        h = self.net(X)

        # Compute test loss
        loss = self.criterion(res, torch.zeros_like(res)).item()
        print(f'Test loss: {loss:.4e}')

        
        # Plot 3D surface
        fig = show_surface(X, h, seed_k, outpath=outpath)

        # Plot contour map for comparing the results between MODFLOW and GW-PINN
        fig = show_contours_2x2(testset, h, seed_k ,outpath=outpath)

        # Plot the error figure between MDOFLOW and GW-PINN
        fig = plot_prediction(testset, h, seed_k, outpath=outpath)

        plt.show()

        # Compute the end time error
        for i, t_ in enumerate(t):
            df = pd.read_csv(f'../MODFLOW/variance=1/seed={seed_k}_{t_}.TXT',
                            delim_whitespace=True,names=['x', 'y', 'h'])
                
            xy = df[['x', 'y']].values 
            t1 = np.ones((len(df[['x']].values),1))*t_
            xyt = np.hstack((xy, t1))
            np.random.seed(seed_k)
            kesi = np.random.randn(1,20)
            xyt_kesi = tile(xyt,kesi)
            xyt_kesi = torch.from_numpy(xyt_kesi).float()
            xyt_kesi = xyt_kesi.to(self.device)
            h_pred = self.net(xyt_kesi)
            h_pred = h_pred.detach().cpu().numpy()
            h = df[['h']].values
                #print(np.hstack((h, h_pred)))

            print(
                f'All area(t={t_}d): mae = {mae(h, h_pred):.3f}, rrmse = {rrmse(h, h_pred) * 100:.3f} %, nse = {nse(h, h_pred):.4f},\n')

        print('Testing finished successfully!!!\n')


    def test_surrogate(self, seed_k, n_logk=200):
    
        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        # load_data
        # seed=200
        # nx, ny, nt = 101, 101, 20
        nx, ny, nt = 51, 51, 20
        # n_logk = 200
        np.random.seed(seed_k)
        kesi = np.zeros((n_logk,20))
        for i_logk in range(n_logk):
            kesi[i_logk,:] = np.random.randn(20)

        x = np.linspace(-500, 500, nx)
        x_ = np.expand_dims(x[1:nx-1], axis=1)
        y = np.linspace(-500, 500, ny)
        y_ = np.expand_dims(y, axis=1)
        xy = tile(x_,y_)
        t_ = 10
        t1 = np.ones(((nx-2)*ny,1))*t_
        xyt = np.hstack((xy, t1))
        xyt_kesi = tile(xyt, kesi)

        # h_pred = np.load('./data/h_data_kle_random_seed=200_n_log=200.npy')
        h_pred = np.load(f'./data/h_data_kle_random_seed={seed_k}_n_log={n_logk}.npy')
        h_pred_ = h_pred[:n_logk, -1, :, 1:nx-1] # x方向的两侧是无流边界,并选取n_logk个实现
        h_ = h_pred_.reshape(-1,1)
        xyt_kesi_h_ = np.hstack((xyt_kesi, h_))
        xyt_kesi_h = xyt_kesi_h_.reshape(n_logk,(nx-2)*ny,24)

        # Compute test loss and error
        loss = np.empty(n_logk)
        mae_ = np.empty(n_logk)
        rrmse_ = np.empty(n_logk)
        nse_ = np.empty(n_logk)
        for i_logk in range(n_logk):
            X_h = xyt_kesi_h[i_logk]
            X = X_h[:, :23]
            h = X_h[:, 23].reshape(-1, 1) # 注意修改数组的形状
            X = torch.from_numpy(X).float()
            X = X.to(self.device)
            h_pred = self.net(X)
            res = self.net_pde(X)
            loss[i_logk] = self.criterion(res, torch.zeros_like(res)).item()
            h_pred = h_pred.detach().cpu().numpy() 
            mae_[i_logk] = mae(h, h_pred)
            rrmse_[i_logk] = rrmse(h, h_pred)
            nse_[i_logk] = nse(h, h_pred)

        name = f"{args.hidden_layers}x{args.hidden_neurons}"
        save_dir = os.path.join('.','results',name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(f'./results/{name}/rrmse_200.npy',rrmse_)
        np.save(f'./results/{name}/mae_200.npy',mae_)
        np.save(f'./results/{name}/nse_200.npy',nse_)
        print('Testing finished successfully!!!\n')

    def test_surrogate2(self, seed_k, n_logk=200):
        
        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        # load_data
        # seed=200
        # nx, ny, nt = 101, 101, 20
        nx, ny, nt = 51, 51, 20
        # n_logk = 200
        np.random.seed(seed_k)
        kesi = np.zeros((n_logk,20))
        for i_logk in range(n_logk):
            kesi[i_logk,:] = np.random.randn(20)

        x = np.linspace(-500, 500, nx)
        x_ = np.expand_dims(x[1:nx-1], axis=1)
        y = np.linspace(-500, 500, ny)
        y_ = np.expand_dims(y, axis=1)
        xy = tile(x_,y_)
        t_ = 10
        t1 = np.ones(((nx-2)*ny,1))*t_
        xyt = np.hstack((xy, t1))
        xyt_kesi = tile(xyt, kesi)

        # h_pred = np.load('./data/h_data_kle_random_seed=200_n_log=200.npy')
        h_pred = np.load(f'./data/h_data_kle_random_seed={seed_k}_n_log={n_logk}.npy')
        h_pred_ = h_pred[:n_logk, -1, :, 1:nx-1] # x方向的两侧是无流边界,并选取n_logk个实现
        h_ = h_pred_.reshape(-1,1)
        xyt_kesi_h_ = np.hstack((xyt_kesi, h_))
        xyt_kesi_h = xyt_kesi_h_.reshape(n_logk,(nx-2)*ny,24)

        # Compute test loss and error
        loss = np.empty(n_logk)
        mae_ = np.empty(n_logk)
        rrmse_ = np.empty(n_logk)
        nse_ = np.empty(n_logk)
        for i_logk in range(n_logk):
            X_h = xyt_kesi_h[i_logk]

            # 剔除井附近-40m~40m的井单元，再计算下列指标
            pos_more = np.where((abs(X_h[:, 0]) > 40) |
                                (abs(X_h[:, 1]) > 40))

            X_h2 = X_h[pos_more]
            X = X_h2[:, :23]
            h = X_h2[:, 23].reshape(-1, 1) # 注意修改数组的形状
            X = torch.from_numpy(X).float()
            X = X.to(self.device)
            h_pred = self.net(X)
            res = self.net_pde(X)
            loss[i_logk] = self.criterion(res, torch.zeros_like(res)).item()
            h_pred = h_pred.detach().cpu().numpy() 
            mae_[i_logk] = mae(h, h_pred)
            rrmse_[i_logk] = rrmse(h, h_pred)
            nse_[i_logk] = nse(h, h_pred)

        name = f"{args.hidden_layers}x{args.hidden_neurons}"
        save_dir = os.path.join('.','results',name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(f'./results/{name}/rrmse_200_4.npy',rrmse_)
        np.save(f'./results/{name}/mae_200_4.npy',mae_)
        np.save(f'./results/{name}/nse_200_4.npy',nse_)
        print('Testing finished successfully!!!\n')

    def test_surrogate3(self, seed_k, n_logk=200):
        
        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        # load_data
        # seed=200
        # nx, ny, nt = 101, 101, 20
        nx, ny, nt = 51, 51, 20
        # n_logk = 200
        np.random.seed(seed_k)
        kesi = np.zeros((n_logk,20))
        for i_logk in range(n_logk):
            kesi[i_logk,:] = np.random.randn(20)

        # Compute test loss and error
        # loss = np.empty((nt, n_logk))
        mae_ = np.empty((nt, n_logk))
        rrmse_ = np.empty((nt, n_logk))
        nse_ = np.empty((nt, n_logk))

        # h_pred = np.load('./data/h_data_kle_random_seed=200_n_log=200.npy')
        h_ref = np.load(f'data/h_data_kle_random_seed={seed_k}_n_log={n_logk}.npy')

        
        x = np.linspace(-500, 500, nx)
        x_ = np.expand_dims(x[1:nx-1], axis=1)
        y = np.linspace(-500, 500, ny)
        y_ = np.expand_dims(y, axis=1)
        xy = tile(x_,y_)
        t = np.linspace(0, 10, nt+1)
        for i in range(nt):
            t_ = t[1+i]
            t1 = np.ones(((nx-2)*ny,1))*t_
            xyt = np.hstack((xy, t1))
            xyt_kesi = tile(xyt, kesi)
            h_ref_ = h_ref[:n_logk, i, :, 1:nx-1] # x方向的两侧是无流边界,并选取n_logk个实现
            h_ = h_ref_.reshape(-1,1)
            xyt_kesi_h_ = np.hstack((xyt_kesi, h_))
            xyt_kesi_h = xyt_kesi_h_.reshape(n_logk,(nx-2)*ny,24)
   
            for i_logk in range(n_logk):
                X_h = xyt_kesi_h[i_logk]

                # 剔除井附近-40m~40m的井单元，再计算下列指标
                pos_more = np.where((abs(X_h[:, 0]) > 40) |
                                    (abs(X_h[:, 1]) > 40))

                X_h2 = X_h[pos_more]
                X = X_h2[:, :23]
                h = X_h2[:, 23].reshape(-1, 1) # 注意修改数组的形状
                X = torch.from_numpy(X).float()
                X = X.to(self.device)
                h_pred = self.net(X)
                # res = self.net_pde(X)
                # loss[i,i_logk] = self.criterion(res, torch.zeros_like(res)).item()
                h_pred = h_pred.detach().cpu().numpy() 
                mae_[i,i_logk] = mae(h, h_pred)
                rrmse_[i,i_logk] = rrmse(h, h_pred)
                nse_[i,i_logk] = nse(h, h_pred)

        name = f"{args.hidden_layers}x{args.hidden_neurons}_{args.nt}"
        save_dir = os.path.join('.','results',name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # np.save(f'./results/{name}/loss_200_2.npy',loss)
        np.save(f'results/{name}/rrmse_200_3.npy',rrmse_)
        np.save(f'results/{name}/mae_200_3.npy',mae_)
        np.save(f'results/{name}/nse_200_3.npy',nse_)

        print('200 Realization testing finished successfully!!!\n')

    def test_error(self, number, nx=99, ny=101):
        
        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        # uq_test
        xyt_kesi_h = np.load(f'./data/test{number}.npy')
        X = xyt_kesi_h[:, :23]
        h = xyt_kesi_h[:, 23].reshape(-1,1)
        X = torch.from_numpy(X).float()
        X = X.to(self.device)
        h_pred = self.net(X)
        res = self.net_pde(X)

        # Compute test loss
        loss = self.criterion(res, torch.zeros_like(res)).item()
        print(f'Test loss: {loss:.4e}')
 
        # h_pred = h_pred.detach().cpu().numpy()  
        # Plot 3D surface
        fig = show_surface(X, h_pred, nx=nx, ny=ny)

        # Plot contour map for comparing the results between MODFLOW and GW-PINN
        fig = show_contours_2x2_2(h_pred, h,nx=nx, ny=ny)

        # Plot the error figure between MDOFLOW and GW-PINN
        # fig = plot_prediction(testset, h, seed_k)

        plt.show()

        print('Testing finished successfully!!!\n')


    def test_uq(self, n_logk=10000,nx=101,ny=101):
        
        t_start = time.time() # 记录开始时间
        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        # uq_test
        # nx, ny, nt = 101, 101, 20
        # h_total = np.empty((n_logk,4,nx,ny),dtype=np.float32)
        h_total = np.empty((n_logk,4,nx,ny))
        kesi = np.zeros((n_logk,20))
        for i_logk in range(n_logk):
            kesi[i_logk,:] = np.random.randn(20)

        x = np.linspace(-500, 500, nx)
        x_ = np.expand_dims(x, axis=1)
        y = np.linspace(-500, 500, ny)
        y_ = np.expand_dims(y, axis=1)
        xy = tile(x_,y_)
        t = [2.5, 5, 7.5, 10]
        t = t = np.array(t)
        xyt = tile(xy,t)
        n = xyt.shape[0]
        for i_logk in range(n_logk):
            kesi1 = kesi[i_logk].reshape(1,-1)
            xyt_kesi = tile(xyt, kesi1)
            X = torch.from_numpy(xyt_kesi).float()
            X = X.to(self.device)
            h_pred = self.net(X)
            h_pred = h_pred.detach().cpu().numpy()
            h0 = h_pred[:n//4, :]
            h1 = h_pred[n//4:n//2, :]
            h2 = h_pred[n//2:3*n//4, :]
            h3 = h_pred[3*n//4:, :]
            h_total[i_logk,0,:,:] = h0.reshape(ny, nx)
            h_total[i_logk,1,:,:] = h1.reshape(ny, nx)
            h_total[i_logk,2,:,:] = h2.reshape(ny, nx)
            h_total[i_logk,3,:,:] = h3.reshape(ny, nx)

        name = f"{args.hidden_layers}x{args.hidden_neurons}"
        # 获得上一级目录的绝对路径
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        save_dir = os.path.join(parent_dir,'UQ_task','base_series',name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(f'../UQ_task/base_series/{name}/h_pred_{n_logk}.npy',h_total)
        t_end = time.time()
        time_consume = t_end - t_start
        print(f'{n_logk} tests total consume {time_consume:.2f} s!!!')

        print('UQ Testing finished successfully!!!\n')


if __name__ == '__main__':

    args = Options().parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.problem = Problem(sigma=args.sigma)
    print('****************************************************************')
    print(f'Unconfined Aquifer Containing A Single Well With Variable Conductivity')
    print(f'domain={args.problem.domain}')
    print(f'sigma={args.sigma}')
    print(f'layers={args.layers}')
    print('****************************************************************')
    tester = Tester(args)

    # tester.test(5) 
    # tester.test(10)
    # tester.test(38) 
    # tester.test(100)
    # tester.test(125)
    # tester.test(500)
    # tester.test_surrogate(5,n_logk=30) # 不确定量化
    # tester.test_surrogate(200, n_logk=200)
    tester.test_surrogate3(200, n_logk=200)
    # tester.test_uq(n_logk=15000, nx=51, ny=51)
    # trainer.test_error(131)
    # trainer.test_error(693)
    # trainer.test_error(693,nx=49,ny=51) # 误差很大
    # trainer.test_error(3358)
    # trainer.test_error(3435) 
    # trainer.test_error(3467)
    # trainer.test_error(3485)  
    # trainer.test_error(3467,nx=39,ny=41) # 误差很大
    # trainer.test_error(3485,nx=49,ny=51) # 误差很大
    # trainer.test_error(4238)
    # trainer.test_error(4654)    