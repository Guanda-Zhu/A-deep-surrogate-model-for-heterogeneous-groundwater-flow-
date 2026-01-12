#!/usr/bin/env python
from options import Options
from utils import save_checkpoints, show_surface, show_contours_2x2, show_contours_2x2_2, plot_prediction, mae, mse, rrmse, nse, tile
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


class Trainer():
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

    def train_info(self, name, epoch, train_loss, valid_loss, tt):
        result = f'{name:5s} '
        result += f'{epoch+1:5d}/{self.epochs_Adam+self.epochs_LBFGS:5d} '
        result += f'train_loss: {train_loss:.4e} '
        result += f'valid_loss: {valid_loss:.4e} '
        result += f'time: {time.time()-tt:5.2f} '
        if name == 'Adam':
            result += f'lr: {self.lr_scheduler.get_last_lr()[0]:.2e}'
        print(result)

    def train(self):

        # Hyperparameters Setting
        self.epochs_Adam = self.args.epochs_Adam
        self.epochs_LBFGS = self.args.epochs_LBFGS
        self.lam = self.args.lam
        self.seed_k = self.args.seed_k

        # Trainset
        self.trainset = Trainset(
            self.problem, nt=args.nt, n_kesi=args.n_kesi, fname='./data/well.mat')
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.args.batch_size, shuffle=True)

        # Validset
        t_valid = [2.5, 5, 7.5, 10]
        kesi_seed = self.seed_k
        #np.random.seed(self.seed_k)
        #kesi_valid = np.random.randn(1,20)
        self.validset = Validset(self.problem, 100, 100, t_valid, kesi_seed)

        self.step = 0

        # Writer
        self.writer = SummaryWriter(comment=f'_{self.model_name}')

        self.train_Adam()
        self.train_LBFGS()

        self.writer.close()

        print('Training finished successfully!!!\n')

    def train_Adam(self):
        """
        Training process using Adam optimizer
        """

        # Optimizer
        optimizer = optim.Adam(self.pinn.parameters(), lr=self.args.lr)
        self.lr_scheduler = StepLR(optimizer, step_size=2000, gamma=0.7)

        best_loss = 1.0e10
        tt = time.time()

        # Training
        # Stage1: Training Process using Adam Optimizer
        for epoch in range(self.epochs_Adam):
            self.pinn.train()
            for data in self.trainloader:

                # Dataset
                X, res, X_bdy2, hx_bdy2 = data
                X = X.view(-1, 23).float().to(self.device)
                res = res.view(-1, 1).float().to(self.device)
                X_bdy2 = X_bdy2.view(-1, 23).float().to(self.device)
                hx_bdy2 = hx_bdy2.view(-1, 1).float().to(self.device)

                optimizer.zero_grad()

                res_pred, hx_bdy2_pred = self.pinn(X, X_bdy2)

                loss = self.criterion(res_pred, res)
                loss_bdy2 = self.criterion(hx_bdy2_pred, hx_bdy2)

                loss_total = loss + self.lam * loss_bdy2
                loss_total.backward()
                optimizer.step()
                self.lr_scheduler.step()
                train_loss = loss_total.item()

            self.writer.add_scalar('train_loss', train_loss, epoch)

            if (epoch + 1) % 100 == 0:
                self.step += 1
                valid_loss = self.validate()

                self.train_info('Adam', epoch, train_loss, valid_loss, tt)
                tt = time.time()

                self.pinn.train()

                is_best = valid_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoints(state, is_best, save_dir=self.model_name)
                if is_best:
                    self.best_loss = valid_loss

    def train_LBFGS(self):
        """
        Training process using LBFGS optimizer
        """
        optimizer = optim.LBFGS(self.pinn.parameters(),
                                max_iter=20,
                                tolerance_grad=1e-8,
                                tolerance_change=1e-12)
        best_loss = 1.0e10
        tt = time.time()

        for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
            self.pinn.train()
            for data in self.trainloader:
                # Dataset
                X, res, X_bdy2, hx_bdy2 = data
                X = X.view(-1, 23).float().to(self.device)
                res = res.view(-1, 1).float().to(self.device)
                X_bdy2 = X_bdy2.view(-1, 23).float().to(self.device)
                hx_bdy2 = hx_bdy2.view(-1, 1).float().to(self.device)

                # Forward and backward propogate
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()

                    res_pred, hx_bdy2_pred = self.pinn(X, X_bdy2)

                    loss = self.criterion(res_pred, res)
                    loss_bdy2 = self.criterion(hx_bdy2_pred, hx_bdy2)

                    loss_total = loss + self.lam * loss_bdy2

                    if loss_total.requires_grad:
                        loss_total.backward()

                    return loss_total

                optimizer.step(closure)
                train_loss = closure().item()
                self.writer.add_scalar('train_loss', train_loss, epoch)

                if math.isnan(train_loss):
                    print('Train_loss is nan! End!!!')
                    break

            if (epoch+1) % 2 == 0:
                self.step += 1
                valid_loss = self.validate()
                self.train_info('LBFGS', epoch, train_loss, valid_loss, tt)
                tt = time.time()

                self.pinn.train()

                is_best = valid_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoints(state, is_best, save_dir=self.model_name)
                if is_best:
                    self.best_loss = train_loss

            if math.isnan(train_loss):
                print('Train_loss is nan! End!!!')
                break
        
    def validate(self):
        """Validate process"""

        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()

        X = self.validset()
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            X = X.to(self.device)

        res = self.net_pde(X)

        loss = self.criterion(res, torch.zeros_like(res))
        valid_loss = loss.item()
        self.writer.add_scalar('valid_loss', valid_loss, self.step)

        # plot
        h = self.net(X)
        """
        fig = show_surface(X, h)
        self.writer.add_figure(
            tag='3D surface', figure=fig, global_step=self.step)

        fig = show_contours_2x2(self.validset, h, 5)
        self.writer.add_figure(tag='contour', figure=fig,
                               global_step=self.step)

        fig = plot_prediction(self.validset, h, 5)
        self.writer.add_figure(tag='Error', figure=fig,
                               global_step=self.step)
        """

        return valid_loss

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
        fig = show_surface(X, h)

        # Plot contour map for comparing the results between MODFLOW and GW-PINN
        fig = show_contours_2x2(testset, h, seed_k)

        # Plot the error figure between MDOFLOW and GW-PINN
        fig = plot_prediction(testset, h, seed_k)

        plt.show()

        # Compute the end time error
        for i, t_ in enumerate(t):
            df = pd.read_csv(f'./MODFLOW/variance=1//seed={seed_k}_{t_}.TXT',
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
            print(
                f'All area(t={t_}d): mae = {mae(h, h_pred):.3f}, rrmse = {rrmse(h, h_pred) * 100:.3f} %, nse = {nse(h, h_pred):.4f}\n')
    

if __name__ == '__main__':
    
    args = Options().parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.problem = Problem(sigma=args.sigma)
    print('****************************************************************')
    print(f'Unconfined Aquifer Containing A Single Well With Variable Conductivity')
    print(f'domain={args.problem.domain}')
    print(f'nt={args.nt}')
    print(f'sigma={args.sigma}')
    print(f'layers={args.layers}')
    print('****************************************************************')
    trainer = Trainer(args)

    trainer.train() # 训练模型
    #trainer.test(5)