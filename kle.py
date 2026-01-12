# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:00:37 2022

@author: Ma Qianxi
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
from pyDOE import lhs
from torch.autograd import Variable

class KLE():
    def __init__(self,L_x,L_y,eta_x,eta_y,var1,mean_logk,n_eigen,weight,nx,ny,delx,dely):
        self.L_x = L_x
        self.L_y = L_y
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.eta = eta_x
        self.var = var1
        self.mean_logk = mean_logk
        self.n_eigen = n_eigen
        self.weight = weight
        self.domain = self.L_x * self.L_y
        self.nx = nx
        self.ny = ny
        self.delx = delx
        self.dely = dely


    def search(self,f,a,h,n):
    # 定义函数搜获隔根区间
    #%功能：找到发f(x)在区间[a,+∞)上所有隔根区间
    #%输入：f(x):所求方程函数；[a,+∞):有根区间；h:步长；n:所需要区间个数
    #%输出：隔根区间[c,d] 
        c=np.empty((n,1))
        d=np.empty((n,1))
        k=0
        while k<=n-1:
            if f(a)*f(a+h)<=0:
                c[k]=a
                d[k]=a+h
                k=k+1
            a=a+h
        return c,d

    def newton(self,fname,dfname,x0,tol,N,m):
    # 牛顿迭代法及其修改形式
    # 功能：牛顿迭代法及其修改形式
    # 输入：初值x0,最大迭代步数N，误差限tol，m=1为牛顿迭代法，m>1为修改的牛顿迭代法
    # 输出：近似根y，迭代次数k    
        y=x0
        x0=y+2*tol
        k=0
        while np.abs(x0-y)>tol and k<N:
            k=k+1
            x0=y
            y=x0-m*fname(x0)/dfname(x0)
        
        if k==N:
            print("warning")
        
        return y,k

    def eigen_value_solution(self,eta,L):
    # 功能：求解一维特征值 
    # 输入：eta:相关长度   L：区域长度   var:方差   Num_Root: 前N个实根
    # 输出：Lamda：特征值  w0：对应特征方程正实根
        var = self.var
        Num_Root = self.n_eigen   
        w0=np.empty((Num_Root,1))
        lamda=np.empty((Num_Root,1))
        cumulate_lamda=np.empty((Num_Root,1))
        ##############################################################################
        ##定义方程形式
        #########################################################################
        def ff(x):
            ff=(eta**2*x**2-1)*np.sin(x*L)-2*eta*x*np.cos(x*L)
            return ff    
        ##############################################################################
        ##定义方程导数形式
        #########################################################################
        def dff(x):
            dff=(2*eta**2*x-1)*np.sin(x*L)+(eta**2*x**2-1)*np.cos(x*L)*L-2*eta*np.cos(x*L)+2*eta*x*np.sin(x*L)*L
            return dff 
        ##用函数搜索隔根区间
        c,d=self.search(ff,0.00001,0.00001,Num_Root)
        w00=(c+d)/2    
        #%用牛顿法精确求解
        for i in range(Num_Root):
            w0[i],k= self.newton(ff,dff,w00[i],1e-8,10000,1)
        ## 根据特征方程正实根，计算特征值λ（Lamda） %%%%
        for flag in range(Num_Root):
            lamda[flag]=2*eta*var/(eta**2*w0[flag]**2+1)
            if flag==0:
                cumulate_lamda[flag]=lamda[flag]
            else:
                cumulate_lamda[flag]=lamda[flag]+cumulate_lamda[flag-1]       
        return lamda,w0,cumulate_lamda

    def sort_lamda(self,lamda_x,w0_x,lamda_y,w0_y):
    #  功能：二维特征值组合并排序
    #  输入：Lamda_x，w0_x:x方向特征值以及对应特征方程实根，Lamda_y,w0_y:y方向特征值以及对应特征方程实根
    #       Domain：矩形域范围，var：方差
    #       N_X,N_Y:X,Y方向特征值个数
    #  输出：lamda:按递减顺序排列的二维特征值
    #       w_x,w_y:特征值对应特征方程在不同方向上的正实根
    #       n：特征值截断个数，权重weight, cum_lamda:特征根累计值
        domain = self.domain
        var = self.var
        weight = self.weight
        n_x=len(w0_x)
        n_y=len(w0_y)
        num=n_x*n_y
        lamda_2d=np.zeros((num,1))
        flag=0
        lamda_index=list()
        for i in range(n_x):
            for j in range(n_y):
                lamda_2d[flag]=lamda_x[i]*lamda_y[j]/var
                lam_ind=[lamda_2d[flag],i,j]          
                lamda_index.append(lam_ind)
                flag=flag+1    
        lamda_index_sorted=sorted(lamda_index, key = lambda x: x[0], reverse=True)    
        sum_lamda=np.zeros((num,1))
        lamda_all=np.zeros((num,1))
        w_x_all=np.zeros((num,1))
        w_y_all=np.zeros((num,1))
        # sum_lamda[0]=lamda_index_sorted[0][0]    
        lab=1    
        for k in range(num):
            lamda_all[k]=lamda_index_sorted[k][0]
            w_x_all[k]=w0_x[lamda_index_sorted[k][1]]
            w_y_all[k]=w0_y[lamda_index_sorted[k][2]]        
            if k==0:
                sum_lamda[k]=lamda_index_sorted[k][0]
            else:
                sum_lamda[k]=sum_lamda[k-1]+lamda_index_sorted[k][0]           
            if lab and sum_lamda[k]/domain/var>=weight:
                n=k+1
                lab=0       
  
        cum_lamda=np.zeros((n,1))
        lamda=np.zeros((n,1))
        w_x=np.zeros((n,1))
        w_y=np.zeros((n,1))        
        for kk in range(n):
            lamda[kk]=lamda_all[kk]
            w_x[kk]=w_x_all[kk]
            w_y[kk]=w_y_all[kk]
            cum_lamda[kk]=sum_lamda[kk] 
        return lamda,w_x,w_y,n,cum_lamda

    def eigen_func(self,n,w,L,x):
    #功能：计算特征值对应的特征函数值     
    #输入：n:特征值截断个数  w：特征方程正实根  eta:相关长度    L:区域长度   x:位置
    #输出：f:特征值对应的特征函数值
        eta = self.eta
        f=np.empty((n,1))
        for i in range(n):
            f[i]=(eta*w[i]*np.cos(w[i]*x)+np.sin(w[i]*x))/np.sqrt((eta**2*w[i]**2+1)*L/2+eta)
        return f

    def eigen_func2(self,n,w,eta,L,x):
    #计算特征值对应的特征函数值     
    #输入：   n:特征值截断个数  w：特征方程正实根  eta:相关长度    L:区域长度   x:位置
    #输出：   f:特征值对应的特征函数值
        #eta = self.eta
        f=torch.empty((n,len(x)))
        for i in range(n):
            f[i,:]=(eta*w[i]*torch.cos(w[i]*x)+torch.sin(w[i]*x))/torch.sqrt((eta**2*w[i]**2+1)*L/2+eta)
        return f

    def plot(self):
        '''
        plot the eigenvalues distribution
        '''
        nx = self.nx
        ny = self.ny
        delx = self.delx
        dely = self.dely
        x=np.arange(1,nx+1,1)
        x=x*delx
        y=np.arange(1,ny+1,1)
        y=y*dely
        n_eigen = self.n_eigen

        lamda_x,w_x0,cumulate_lamda_x = self.eigen_value_solution(self.eta_x,self.L_x)
        lamda_y,w_y0,cumulate_lamda_y = self.eigen_value_solution(self.eta_y,self.L_y)
        fig, ax1 = plt.subplots()
        ax1.plot(range(1,n_eigen+1),lamda_x/self.L_x,label='eta=408, lamda_x/L')
        ax1.plot(range(1,n_eigen+1),lamda_y/self.L_y,label='eta=408, lamda_y/L')
        ax1.set_xlim([1,n_eigen])
        plt.legend()
        ax1.set_xlabel('n')
        ax1.set_ylabel('lamda/L')
        fig, ax = plt.subplots()
        ax.plot(range(1,n_eigen+1),cumulate_lamda_x/self.L_x,label='eta=408, sum_lamda_x/L')
        ax.plot(range(1,n_eigen+1),cumulate_lamda_y/self.L_y,label='eta=408, sum_lamda_y/L')
        ax.set_xlim([1,n_eigen])
        plt.legend()
        ax.set_xlabel('n')
        ax.set_ylabel('sum_lamda/L')
        plt.show()


    def gen_field(self, seed_n, n_logk):
        '''
        generate a heterogeneous hydraulic conductivity field
        '''
        nx = self.nx
        ny = self.ny
        delx = self.delx
        dely = self.dely
        x=np.arange(1,nx+1,1)
        x=x*delx
        y=np.arange(1,ny+1,1)
        y=y*dely

        lamda_x,w_x0,cumulate_lamda_x = self.eigen_value_solution(self.eta_x,self.L_x)
        lamda_y,w_y0,cumulate_lamda_y = self.eigen_value_solution(self.eta_y,self.L_y)
        lamda_xy,w_x,w_y,n,cum_lamda=self.sort_lamda(lamda_x,w_x0,lamda_y,w_y0)
        n_eigen=n
        fn_x=[]
        fn_y=[]
        for i_x in range(self.nx):
            f_x=self.eigen_func(n_eigen, w_x, self.L_x, x[i_x])
            fn_x.append([f_x,x[i_x]])  
        for i_y in range(self.ny):
            f_y=self.eigen_func(n_eigen, w_y, self.L_y, y[i_y])
            fn_y.append([f_y,y[i_y]])
        #生成随机数组，生成渗透率场实现    
        np.random.seed(seed_n)
        kesi=np.zeros((n_logk,n_eigen))   #随机数数组
        logk=np.empty((self.nx,self.ny))       #渗透率场数组
        for i_logk in range(n_logk):
            kesi[i_logk,:]=np.random.randn(n_eigen)   #随机数数组 
            #由随机数计算渗透率场
            for i_x in range(self.nx):
                for i_y in range(self.ny):
                    logk[i_y,i_x]=self.mean_logk+np.sum(np.sqrt(lamda_xy)*fn_x[i_x][0]*fn_y[i_y][0]*kesi[i_logk:i_logk+1].transpose())
                    #logk[i_logk,i_y,i_x]=self.mean_logk
                    #for i_eigen in range(n_eigen):
                        #logk[i_y,i_x]=logk[i_y,i_x]+np.sqrt(lamda_xy[i_eigen])*fn_x[i_x][0][i_eigen]*fn_y[i_y][0][i_eigen]*kesi[i_logk,i_eigen]
        k=np.exp(logk)
        return logk, k, n_eigen


    def gen_field2(self, seed_n, n_logk):
        '''
        generate many heterogeous hydraulic conductivity fields
        '''
        nx = self.nx
        ny = self.ny
        delx = self.delx
        dely = self.dely
        x=np.arange(1,nx+1,1)
        x=x*delx
        y=np.arange(1,ny+1,1)
        y=y*dely

        lamda_x,w_x0,cumulate_lamda_x = self.eigen_value_solution(self.eta_x,self.L_x)
        lamda_y,w_y0,cumulate_lamda_y = self.eigen_value_solution(self.eta_y,self.L_y)
        lamda_xy,w_x,w_y,n,cum_lamda=self.sort_lamda(lamda_x,w_x0,lamda_y,w_y0)
        n_eigen=n
        fn_x=[]
        fn_y=[]
        for i_x in range(self.nx):
            f_x=self.eigen_func(n_eigen, w_x, self.L_x, x[i_x])
            fn_x.append([f_x,x[i_x]])  
        for i_y in range(self.ny):
            f_y=self.eigen_func(n_eigen, w_y, self.L_y, y[i_y])
            fn_y.append([f_y,y[i_y]])
        #生成随机数组，生成渗透率场实现    
        np.random.seed(seed_n)
        kesi=np.zeros((n_logk,n_eigen))   #随机数数组
        logk=np.zeros((n_logk,self.nx,self.ny))       #渗透率场数组
        for i_logk in range(n_logk):
            kesi[i_logk,:]=np.random.randn(n_eigen)   #随机数数组 
            #由随机数计算渗透率场
            for i_x in range(self.nx):
                for i_y in range(self.ny):
                    logk[i_logk,i_y,i_x]=self.mean_logk
                    for i_eigen in range(n_eigen):
                        logk[i_logk,i_y,i_x]=logk[i_logk,i_y,i_x]+np.sqrt(lamda_xy[i_eigen])*fn_x[i_x][0][i_eigen]*fn_y[i_y][0][i_eigen]*kesi[i_logk,i_eigen]
        k=np.exp(logk)
        k_image = k.reshape(n_logk,1,self.nx,self.ny) # 按照图片的形式输出
        return logk, k_image, n_eigen


    def gen_field_tensor(self, xyt):
        #渗透率计算函数，利用torch的方式，生成渗流场的数组（tensor）
        nx = self.nx
        ny = self.ny
        delx = self.delx
        dely = self.dely
        x=np.arange(1,nx+1,1)
        x=x*delx
        y=np.arange(1,ny+1,1)
        y=y*dely
        mean_logk = self.mean_logk

        lamda_x,w_x0,cumulate_lamda_x = self.eigen_value_solution(self.eta_x,self.L_x)
        lamda_y,w_y0,cumulate_lamda_y = self.eigen_value_solution(self.eta_y,self.L_y)
        lamda_xy,w_x,w_y,n,cum_lamda=self.sort_lamda(lamda_x,w_x0,lamda_y,w_y0)
        n_eigen = n
        w_x_tensor = torch.from_numpy(w_x)
        w_x_tensor = w_x_tensor.type(torch.FloatTensor)
        w_y_tensor = torch.from_numpy(w_y)
        w_y_tensor = w_y_tensor.type(torch.FloatTensor)
        lamda_xy_tensor = torch.from_numpy(lamda_xy)
        lamda_xy_tensor = lamda_xy_tensor.type(torch.FloatTensor)
        xyt = torch.tensor(xyt)
        x_tensor = xyt[:,0] + 505
        y_tensor = xyt[:,1] + 505 
        x_tensor = Variable(x_tensor, requires_grad=True)
        y_tensor = Variable(y_tensor, requires_grad=True)
        fx = self.eigen_func2(n_eigen, w_x_tensor, self.eta_x, self.L_x, x_tensor)
        fy = self.eigen_func2(n_eigen, w_y_tensor, self.eta_y, self.L_y, y_tensor)
        #生成随机数组，生成渗透率场实现  
        np.random.seed(self.seed_n)
        n_xyt = len(xyt) # the number of collocation point 
        kesi_n=np.random.randn(n_eigen)   # 随机数数组
        kesi_tensor = torch.tensor(kesi_n) # 转换数据类型
        # for i_logk in range(self.n_logk):
        kesi=torch.empty((n_xyt, n_eigen))
        for i in range(n_eigen):
            kesi[:, i]=kesi_tensor[i]   # 随机数数组再赋值 
        
        fx = fx.transpose(0, 1)
        fy = fy.transpose(0, 1)
        lamda_xy_tensor = lamda_xy_tensor.transpose(0, 1)

        logk=mean_logk+torch.sum(torch.sqrt(lamda_xy_tensor)*kesi*fx*fy,1)
        kk=torch.exp(logk)
        k_x = torch.autograd.grad(outputs=kk.sum(), inputs=x_tensor, \
                          create_graph=True,allow_unused=True)[0].view(n_xyt, 1).detach()

        k_y = torch.autograd.grad(outputs=kk.sum(), inputs=y_tensor, \
                          create_graph=True,allow_unused=True)[0].view(n_xyt, 1).detach()
        #kk.detach()
        kk = kk.detach().numpy().reshape((n_xyt, 1))
        return kk, logk, k_x, k_y

    def gen_field_tensor2(self, xyt_kesi):
        '''根据xyt_kesi数组生成K(x,y)'''
        mean_logk = self.mean_logk

        lamda_x,w_x0,cumulate_lamda_x = self.eigen_value_solution(self.eta_x,self.L_x)
        lamda_y,w_y0,cumulate_lamda_y = self.eigen_value_solution(self.eta_y,self.L_y)
        lamda_xy,w_x,w_y,n,cum_lamda=self.sort_lamda(lamda_x,w_x0,lamda_y,w_y0)
        n_eigen = n
        w_x_tensor = torch.from_numpy(w_x)
        w_x_tensor = w_x_tensor.type(torch.FloatTensor)
        w_y_tensor = torch.from_numpy(w_y)
        w_y_tensor = w_y_tensor.type(torch.FloatTensor)
        lamda_xy_tensor = torch.from_numpy(lamda_xy)
        lamda_xy_tensor = lamda_xy_tensor.type(torch.FloatTensor)

        xyt_kesi = torch.tensor(xyt_kesi)      
        x_tensor = xyt_kesi[:,0] + 510
        y_tensor = xyt_kesi[:,1] + 510 
        kesi = xyt_kesi[:,3:]
        x_tensor = Variable(x_tensor, requires_grad=True)
        y_tensor = Variable(y_tensor, requires_grad=True)
        fx = self.eigen_func2(n_eigen, w_x_tensor, self.eta_x, self.L_x, x_tensor)
        fy = self.eigen_func2(n_eigen, w_y_tensor, self.eta_y, self.L_y, y_tensor)

        fx = fx.transpose(0, 1)
        fy = fy.transpose(0, 1)
        lamda_xy_tensor = lamda_xy_tensor.transpose(0, 1)

        logk=mean_logk+torch.sum(torch.sqrt(lamda_xy_tensor)*kesi*fx*fy,1)
        kk=torch.exp(logk)
        n_xyt = len(xyt_kesi) # the number of collocation point 
        k_x = torch.autograd.grad(outputs=kk.sum(), inputs=x_tensor, \
                          create_graph=True,allow_unused=True)[0].view(n_xyt, 1).detach()

        k_y = torch.autograd.grad(outputs=kk.sum(), inputs=y_tensor, \
                          create_graph=True,allow_unused=True)[0].view(n_xyt, 1).detach()
        #kk.detach()
        kk = kk.detach().numpy().reshape((n_xyt, 1))
        return kk, logk, k_x, k_y

if __name__ == '__main__':
    mean_logk = 0 # 均值
    var1 = 1.0 # 方差

    nx = 101
    ny = 101
    delx = 10
    dely = 10
    L_x = nx * delx #1010 #区域长度
    L_y = ny * dely #1010
    # eta=404   #相关长度
    eta_x = L_x*0.4 # x方向的相关长度
    eta_y = L_y*0.4 # y方向的相关长度

    weight = 0.8 #KL展开需要保存的扰动量，80%
    n_eigen = 50
    seed_n = 38 # 200
    n_logk = 1 # KL展开的实现次数
    #n_logk = 200

    kle = KLE(L_x,L_y,eta_x,eta_y,var1,mean_logk,n_eigen,weight,nx,ny,delx,dely)
    # 随机取点
    # lb = np.array([-500, -500, 0])
    # ub = np.array([500, 500, 10])
    # xyt = lb + (ub-lb)*lhs(3, 2500)

    # def tile(x, y):
    #     X = np.tile(x, (y.shape[0], 1))
    #     Y = np.vstack([np.tile(y[i], (x.shape[0], 1)) for i in range(y.shape[0])])

    #     return np.hstack((X, Y))
    # n_eigen = 20  
    # np.random.seed(seed_n)
    # kesi = np.random.randn(n_eigen).reshape((1,-1))   #随机数数组
    # xyt_kesi = tile(xyt, kesi)
    #kk2,logk2,k_x2,k_y2 = kle.gen_field_tensor2(xyt_kesi)

    #kk,logk,k_x,k_y = kle.gen_field_tensor(xyt) # 调试好
    # logk,k,n = kle.gen_field(seed_n,n_logk) # 渗流场的一次实现
    logk,k_image,n = kle.gen_field2(seed_n,10) # 渗流场的n_log次实现
    print(f'KL展开截断的项数为{n}') # 输出需要的项数

    plt.style.use('classic')
    plt.rcParams['xtick.direction'] = 'out'  # 将x轴的刻度线方向设置向外
    plt.rcParams['ytick.direction'] = 'out'  # 将y轴的刻度方向设置向内外
    plt.title(f'$ln K(x,y)$')
    plt.xlabel(f'$x/m$')
    plt.ylabel(f'$y/m$')
    # plt.imshow(logk, origin='lower', interpolation='nearest', extent=(-500, 500, -500, 500)) # 一次实现画图
    plt.imshow(logk[0], origin='lower', interpolation='nearest', extent=(-500, 500, -500, 500)) # n_log次实现某次画图
    plt.colorbar()
    plt.savefig('./figures/logk_5.jpg', dpi=600)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()