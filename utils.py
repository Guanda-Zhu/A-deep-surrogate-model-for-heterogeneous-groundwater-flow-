#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import shutil
import os


def tile(x, y):
    X = np.tile(x, (y.shape[0], 1))
    Y = np.vstack([np.tile(y[i], (x.shape[0], 1)) for i in range(y.shape[0])])

    return np.hstack((X, Y))

def tile_kesi(xyt, seed):
    np.random.seed(seed)
    kesi = np.random.randn(len(xyt),20)
    return np.hstack((xyt, kesi))


def grad(output, input):
    """find the gradient of output w.r.t. input

    Params
    ======
    output (n, 1) tensor
    input  (n, dim) tensor
        it must be setting as requires_grad_(True) before feeding into the network
    """
    w = torch.ones_like(input[:, [0]])
    df = torch.autograd.grad(output, input, w, create_graph=True)[0]

    return df

def load_data(nx=51, ny=51, nt=20, n_logk=30, seed=5):
    """load the training data 

    Params
    ======
    nx: the number of the x-direction grids
    ny: the number of the y-direction grids
    nt: the number of the time steps for one realization
    n_logk: the number of the realizations
    seed: the random seed for generating the conductivity field
    """
    np.random.seed(seed)
    kesi = np.zeros((n_logk,20))
    for i_logk in range(n_logk):
        kesi[i_logk,:] = np.random.randn(20)

    x = np.linspace(-500, 500, nx)
    x_ = np.expand_dims(x[1:nx-1], axis=1)
    y = np.linspace(-500, 500, ny)
    y_ = np.expand_dims(y, axis=1)
    xy = tile(x_,y_)
    t = np.linspace(0, 4, nt+1)
    t_ = np.expand_dims(t[1:], axis=1)
    xt = tile(xy[:,[0]], t_)
    yt = tile(xy[:,[1]], t_)
    xyt = np.hstack((xt[:, [0]], yt[:, [0]], yt[:, [1]]))
    xyt_kesi = tile(xyt, kesi)

    h_pred = np.load(f'./data/h_data_kle_random_seed={seed}_n_log=30.npy')
    h_pred_ = h_pred[:n_logk, :, :, 1:nx-1] # x方向的两侧是无流边界,并选取n_logk个实现
    h_ = h_pred_.reshape(-1,1)
    xyt_kesi_h_ = np.hstack((xyt_kesi, h_))
    xyt_kesi_h = xyt_kesi_h_.reshape(n_logk*nt,(nx-2)*ny,24)

    return xyt_kesi_h

def show_surface(X, h, nx=100, ny=100):
    """
    X: torch.tensor (:, 4)
    h: torch.tensor (:, 1)
    """
    times = [2.5, 5, 7.5, 10]
    X = X.cpu().numpy()
    h = h.detach().cpu().numpy()

    n = X.shape[0]
    # nx = 100
    # ny = 100

    X0 = X[:n//4, :]
    X1 = X[n//4:n//2, :]
    X2 = X[n//2:3*n//4, :]
    X3 = X[3*n//4:, :]

    h0 = h[:n//4, :]
    h1 = h[n//4:n//2, :]
    h2 = h[n//2:3*n//4, :]
    h3 = h[3*n//4:, :]

    x = []
    x.append(X0[:, 0].reshape(ny, nx))
    x.append(X1[:, 0].reshape(ny, nx))
    x.append(X2[:, 0].reshape(ny, nx))
    x.append(X3[:, 0].reshape(ny, nx))

    y = []
    y.append(X0[:, 1].reshape(ny, nx))
    y.append(X1[:, 1].reshape(ny, nx))
    y.append(X2[:, 1].reshape(ny, nx))
    y.append(X3[:, 1].reshape(ny, nx))

    h = []
    h.append(h0.reshape(ny, nx))
    h.append(h1.reshape(ny, nx))
    h.append(h2.reshape(ny, nx))
    h.append(h3.reshape(ny, nx))

    # Plot
    fig = plt.figure(figsize=(12, 10))
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14

    for i in range(0, 4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        surf = ax.plot_surface(x[i], y[i], h[i],
                               cmap=cm.rainbow, linewidth=0, antialiased=False)
        ax.set_title(f'$t={times[i]}$ d')
        if i == 0:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('h (m)')

        # save data
        # data = np.vstack(
        #     (x[i].reshape(-1), y[i].reshape(-1), h[i].reshape(-1))).T
        # columns = ['x', 'y', 'h']
        # df = pd.DataFrame(columns=columns, data=data)
        # df.to_csv(f'data/t_{times[i]}.csv', index=False)

    # Add a color bar which map values to colors
    cax = fig.add_axes([0.94, 0.2, 0.01, 0.6])
    fig.colorbar(surf, cax=cax)
    # plt.savefig('3D_Figure1.png', dpi=600)

    return fig


def show_contours_2x2(validset, h, seed_k):
    
    times = [2.5, 5, 7.5, 10]
    # times = [0.25, 0.5, 0.75, 1]

    grid_x, grid_y = validset.spatial(grid=True)
    h = h.detach().cpu().numpy()

    n = h.shape[0]

    h0 = h[:n//4, :]
    h1 = h[n//4:n//2, :]
    h2 = h[n//2:3*n//4, :]
    h3 = h[3*n//4:, :]

    h = []
    h.append(h0.reshape(100, 100))
    h.append(h1.reshape(100, 100))
    h.append(h2.reshape(100, 100))
    h.append(h3.reshape(100, 100))

    fig = plt.figure(figsize=(10, 10))
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16

    for i, t in enumerate(times):
        df = pd.read_csv(
            f'../MODFLOW/variance=1/seed={seed_k}_{t}.TXT',
            delim_whitespace=True,
            names=['x', 'y', 'h'])
        points = df[['x', 'y']].values 
        values = df[['h']].values

        grid_h_modflow = griddata(
            points, values, (grid_x, grid_y), method='cubic')[:, :, 0]

        grid_h = h[i]

        ax = fig.add_subplot(2, 2, i+1)
        # ax.grid()
        cs1 = ax.contour(grid_x, grid_y, grid_h_modflow,
                         colors='k',
                         levels = [79, 79.2, 79.4, 79.6, 79.8, 79.9]
                         # levels=[80, 83, 86, 89, 92, 95]
                         )
        cs2 = ax.contour(grid_x, grid_y, grid_h,
                         colors='r',
                         levels = [79, 79.2, 79.4, 79.6, 79.8, 79.9]
                         # levels=[80, 83, 86, 89, 92, 95]
                         )
        h1, _ = cs1.legend_elements()
        h2, _ = cs2.legend_elements()
        ax.legend([h1[0], h2[0]], ['MODFLOW', 'GW-PINN'])
        ax.clabel(cs1, fmt='%.1f', inline=1)
        ax.clabel(cs2, fmt='%.1f', inline=1)
        ax.set_title(f'$t={t}$ d')
        if i == 0:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        if i == 1:
            ax.text(350, 510, 'Unit (m)')

        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.savefig('Contour1.png', dpi=600)

    return fig

def show_contours_2x2_2(h, h_origin, nx=100, ny=100):
    
    times = [2.5, 5, 7.5, 10]
    # times = [0.25, 0.5, 0.75, 1]

    # grid_x, grid_y = validset.spatial(grid=True)
    a = np.linspace(-500, 500, ny)
    b = np.linspace(-490, 490, nx)
    grid_x, grid_y = np.meshgrid(b, a)
    h = h.detach().cpu().numpy()

    n = h.shape[0]

    h0 = h[:n//4, :]
    h1 = h[n//4:n//2, :]
    h2 = h[n//2:3*n//4, :]
    h3 = h[3*n//4:, :]

    h = []
    h.append(h0.reshape(ny, nx))
    h.append(h1.reshape(ny, nx))
    h.append(h2.reshape(ny, nx))
    h.append(h3.reshape(ny, nx))

    n = h_origin.shape[0]

    h0 = h_origin[:n//4, :]
    h1 = h_origin[n//4:n//2, :]
    h2 = h_origin[n//2:3*n//4, :]
    h3 = h_origin[3*n//4:, :]

    h_origin = []
    h_origin.append(h0.reshape(ny, nx))
    h_origin.append(h1.reshape(ny, nx))
    h_origin.append(h2.reshape(ny, nx))
    h_origin.append(h3.reshape(ny, nx))


    fig = plt.figure(figsize=(10, 10))
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16

    for i, t in enumerate(times):
        grid_h_modflow = h_origin[i]
        grid_h = h[i]

        ax = fig.add_subplot(2, 2, i+1)
        # ax.grid()
        cs1 = ax.contour(grid_x, grid_y, grid_h_modflow,
                         colors='k',
                         levels = [79, 79.2, 79.4, 79.6, 79.8, 79.9]
                         # levels=[80, 83, 86, 89, 92, 95]
                         )
        cs2 = ax.contour(grid_x, grid_y, grid_h,
                         colors='r',
                         levels = [79, 79.2, 79.4, 79.6, 79.8, 79.9]
                         # levels=[80, 83, 86, 89, 92, 95]
                         )
        h1, _ = cs1.legend_elements()
        h2, _ = cs2.legend_elements()
        ax.legend([h1[0], h2[0]], ['MODFLOW', 'GW-PINN'])
        ax.clabel(cs1, fmt='%.1f', inline=1)
        ax.clabel(cs2, fmt='%.1f', inline=1)
        ax.set_title(f'$t={t}$ d')
        if i == 0:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        if i == 1:
            ax.text(350, 510, 'Unit (m)')

        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.savefig('Contour1.png', dpi=600)

    return fig

def plot_prediction(validset, h, seed_k, nx=100, ny=100):

    times = [2.5, 5, 7.5, 10]
    #times = [0.25, 0.5, 0.75, 1]
    rows = [r'$t=2.5d$', r'$t=5d$', r'$t=7.5d$', r'$t=10d$']
    #rows = [r'$t=0.25d$', r'$t=0.5d$', r'$t=0.75d$', r'$t=1.0d$']
    
    cols = ['Prediction', 'Reference', r'Prediction $-$ Reference']
    n_times = len(times)

    grid_x, grid_y = validset.spatial(grid=True)
    h = h.detach().cpu().numpy()

    n = h.shape[0]

    h0 = h[:n//4, :]
    h1 = h[n//4:n//2, :]
    h2 = h[n//2:3*n//4, :]
    h3 = h[3*n//4:, :]

    h = []
    h.append(h0.reshape(100, 100))
    h.append(h1.reshape(100, 100))
    h.append(h2.reshape(100, 100))
    h.append(h3.reshape(100, 100))

    fig, axes = plt.subplots(n_times, 3, figsize=(15, 4 * n_times))
    for m, t in enumerate(times):
        df = pd.read_csv(
            f'../MODFLOW/variance=1/seed={seed_k}_{t}.TXT',
            delim_whitespace=True,
            names=['x', 'y',  'h'])
        points = df[['x', 'y']].values 
        values = df[['h']].values

        grid_h_modflow = griddata(
            points, values, (grid_x, grid_y), method='cubic')[:, :, 0]
    
        grid_h = h[m]

        error_h = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                if grid_h_modflow[i,j] == None:
                    error_h[i,j] = None
                else:
                    error_h[i,j] = grid_h[i,j] - grid_h_modflow[i,j]

        ax1 = axes[m, 0]
        ax1.set_aspect('equal')
        # ax.set_axis_off()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.xaxis.set_major_locator(plt.NullLocator()) # 隐藏刻度
        ax1.yaxis.set_major_locator(plt.NullLocator()) # 隐藏刻度
        # cax = ax1.imshow(grid_h, extent=(-500, 500, -500, 500), cmap='jet', origin='upper')   
        cax = ax1.imshow(grid_h, extent=(-500, 500, -500, 500), cmap='rainbow', origin='lower', vmin=79, vmax=80)   
        cbar = plt.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[m, 1]
        ax2.set_aspect('equal')
        # ax.set_axis_off()
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.xaxis.set_major_locator(plt.NullLocator()) 
        ax2.yaxis.set_major_locator(plt.NullLocator()) 
        cax = ax2.imshow(grid_h_modflow, extent=(-500, 500, -500, 500), cmap='rainbow', origin='lower', vmin=79, vmax=80)   
        cbar = plt.colorbar(cax, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = axes[m, 2]
        ax3.set_aspect('equal')
        # ax.set_axis_off()
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.xaxis.set_major_locator(plt.NullLocator())
        ax3.yaxis.set_major_locator(plt.NullLocator())
        # cax = ax3.imshow(error_h, extent=(-500, 500, -500, 500), cmap='jet', origin='upper', vmin=0.0, vmax=1.0) 
        cax = ax3.imshow(error_h, extent=(-500, 500, -500, 500), cmap='rainbow_r', origin='lower')   
        cbar = plt.colorbar(cax, ax=ax3, fraction=0.046, pad=0.04)
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, size='large')

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
        # plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.subplots_adjust(top=0.93)
    # plt.savefig('Error1.png', dpi=600)
    #plt.show()

    return fig


def save_checkpoints(state, is_best=None,
                     base_dir='checkpoints',
                     save_dir=None):
    if save_dir:

        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint, best_model)


# def compare(fname, net, args):
#     df = pd.read_csv(fname,
#                      delim_whitespace=True, names=['x', 'y', 't', 'h'])
#     xyt = df[['x', 'y', 't']].values
#     xyt = torch.from_numpy(xyt).float()

#     if args.device == torch.device(type='cuda', index=args.cuda_index):
#         xyt = xyt.to(args.device)
#     h_pred = net(xyt)
#     h_pred = h_pred.detach().cpu().numpy()
#     h = df[['h']].values
#     print(h)
#     print(h_pred)
#     return mae(h, h_pred), rmse(h, h_pred), rrmse(h, h_pred)


def mae(h, h_pred):
    return np.mean(np.abs(h - h_pred))


def mse(h, h_pred):
    return np.mean(np.square(h - h_pred))

def rmse(h, h_pred):
    return np.sqrt(mse(h, h_pred))

def rrmse(h, h_pred):
    return np.sqrt(mse(h, h_pred) / np.mean(h)**2)

def nse(h, h_pred):
    # 纳什效率系数，nash-sutcliffe efficiency coefficient 
    r2=1-np.sum((np.square(h_pred-h)))/np.sum(np.square(h-np.mean(h)))
    return r2

def R2_score(h, h_pred):
    return np.sum((h-np.mean(h))*(h_pred-np.mean(h_pred)))**2/(np.sum((h-np.mean(h))**2)*np.sum((h_pred-np.mean(h_pred))**2))

def me(h, h_pred):
    # 平均误差
    return np.mean(h_pred)-np.mean(h)

def mape(h, h_pred):
    # 平均绝对误差
    return np.mean(np.abs((h_pred-h)/h))

if __name__ == '__main__':
    x = np.random.rand(5, 2)
    y = np.random.rand(3, 2)
    z = tile(x, y)
    print(z)
