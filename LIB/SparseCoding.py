# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:26:07 2020

@author: 月光下的云海
"""
import numpy as np

'''============================================================================
FUNCTION:shrink
FEATURE: 利用proxmal原理对1范数收缩
INPUTS:
       x----------收缩前变量
       tau--------正则参数
OUTPUTS:
	   y----------收缩后的值
============================================================================'''  
def shrink(x,tau = 1):
    x_abs = np.abs(x)-tau
    x_abs[x_abs<0] = 0
    y = np.sign(x)*x_abs
    return y

'''============================================================================
FUNCTION:ISTA
FEATURE: 利用ISTA进行稀疏编码
INPUTS:
       Dic------字典
       b--------原始向量
       tau------正则参数
       step-----步长
OUTPUTS:
	   x--------近似解
============================================================================'''  
def ISTA(Dic,b,tau,step = 1):
    N = max(b.shape)
    x = np.ones((N,1))
    b = np.reshape(b,(N,1))
    grad = Dic.T @ (Dic @ x - b)
    x = x-step*grad
    x = shrink(x)
    return x

'''============================================================================
FUNCTION:lsomp
FEATURE: 用LSOMP算法对向量进行稀疏表示
INPUTS:
   A-------测量矩阵
   b-------测量后的向量
   K-------稀疏度
OUTPUTS:
   稀疏向量hat_x
============================================================================'''  
def lsomp(A,b,K):
    N = A.shape[1]
    hat_x = np.zeros((N,1));hat_x = np.matrix(hat_x)
    r = b
    S = [];Sc = [i for i in range(0,N)]
    E = []
    for i in Sc:
        Ei = np.linalg.norm((np.transpose(A[:,i])*b)[0,0]*A[:,i]/(np.linalg.norm(A[:,i])**2)-b)
        E.append(Ei)
    pos = E.index(min(E))
    i0 = Sc[pos];Sc.remove(i0)
    x = np.transpose(A[:,i0])*b/np.linalg.norm(A[:,i0])**2
    S.append(i0)
    r = b-A[:,S]*x
    for i in range(1,K):
        E = []
        invM = (np.transpose(A[:,S])*A[:,S]).I
        Asc = A[:,Sc]
        Corr = np.squeeze(np.array(Asc.T*r))
        E = list(Corr)
        pos = E.index(max(E))
        i0 = Sc[pos]
        Sc.remove(i0)
        c = np.linalg.norm(A[:,i0])
        bb = np.transpose(A[:,S])*A[:,i0]
        pp = 1/(c-np.transpose(bb)*invM*bb);
        p = pp[0,0]
        temp = np.hstack([invM+p*invM*bb*np.transpose(bb)*invM,-p*invM*bb])
        temp2 = np.hstack([-p*np.transpose(bb)*invM,pp])
        temp = np.vstack([temp,temp2])
        x = temp*np.vstack([np.transpose(A[:,S])*b,np.transpose(A[:,i0])*b])
        S.append(i0)
        r = b-A[:,S]*x
    hat_x[S] = x
    return hat_x
















