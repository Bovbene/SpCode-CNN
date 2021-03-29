# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:34:01 2020

@author: 月光下的云海
"""
from LIB.ClassBlock_image import Block_image
import datetime
import numpy as np
from sklearn.cluster import KMeans
from LIB.ClassDictionary import Dictionary
from LIB.utils import psnr
from LIB.SparseCoding import ISTA,lsomp
from .ClassBlock_image import BlockImage


'''============================================================================
FUNCTION:pca_ISTA
FEATURE: 用PCA字典学习+ISTA编码做字典学习
INPUTS:image,size_block,K,flag
   image---------原始图像
   size_block----图像块大小
   K-------------聚类个数
   flag----------是否显示时间
OUTPUTS:N_image,bloc_im_li,dic_li
   N_image-------重建之后的图像
   bloc_im_li----图像块列表(其中包含稀疏编码)
   dic_li--------字典列表
============================================================================'''
def PCAnISTA(Y,K,blk_size = None):
    dic_li = []
    block_li = []
    if blk_size is None:
        print('Warning: Cannot calucate sub-images')
    for i in range(Y.shape[1]):
        p = Y[:,i].reshape((-1,1))
        if blk_size is not None:
            sub_image = p.reshape((blk_size,blk_size),order = 'F')
        else:
            sub_image = None
        row_stretch = p
        block_li += [BlockImage(sub_image,row_stretch)]
    
    cluster = KMeans(n_clusters = K)#构造聚类器
    cluster.fit(Y.T)#聚类
    label_pred = cluster.labels_ #获取聚类标签
    centroids = cluster.cluster_centers_.T #获取聚类中心
    
    for i,idx in enumerate(label_pred):
        block_li[i].idx = idx
    
    for i in range(K):
        idx = np.where(label_pred == i)[0][:]
        Ck = Y[:,idx]
        Omega = np.cov(Ck)
        dic,D,V = np.linalg.svd(Omega)
        dic_li.append(Dictionary(dic,i,centroids[:,i]))
        for id in idx:
            b = np.array(Y[:,id])
            rec = ISTA(dic,b,tau = 1)
            block_li[id].sparse_code = rec
    
    n_Y = np.zeros(Y.shape)
    for i in range(Y.shape[1]):
        sp_code = block_li[i].sparse_code
        idx = block_li[i].idx
        dic = dic_li[idx].dic
        n_Y[:,i] = np.squeeze(dic @ sp_code)
    return dic_li,block_li,n_Y,label_pred

'''============================================================================
FUNCTION:pca_ISTA
FEATURE: 用PCA字典学习+ISTA编码做字典学习
INPUTS:image,size_block,K,flag
   image---------原始图像
   size_block----图像块大小
   K-------------聚类个数
   flag----------是否显示时间
OUTPUTS:N_image,bloc_im_li,dic_li
   N_image-------重建之后的图像
   bloc_im_li----图像块列表(其中包含稀疏编码)
   dic_li--------字典列表
============================================================================'''
def pca_ISTA(image,size_block,K,flag = True):

    m = image.shape[0]
    #分块
    num_block = int(m/size_block)#分块个数
    bloc_im_li = [[] for i in range(0,num_block)]#分块图像矩阵
    for i in range(0,num_block):
        for j in range(0,num_block):
            sub_image = image[size_block*i:size_block*i+size_block,size_block*j:size_block*j+size_block]
            block_image = Block_image(i,j,sub_image,size_block)
            bloc_im_li[i].append(block_image)
      
    cur1 = datetime.datetime.now()
    if(flag):
        print('Begin at:  '+str(cur1))
        

    #*************************** KNN聚类 **********************************
    dic_li = []
    Y = np.matrix(np.zeros((size_block*size_block,num_block*num_block)))
    for i in range(0,num_block):
        for j in range(0,num_block):
            Y[:,num_block*i+j] = bloc_im_li[i][j].row_stretch
    c_Y = np.transpose(Y)
    cluster0 = KMeans(n_clusters = K)#构造聚类器
    cluster0.fit(c_Y)#聚类
    label_pred = cluster0.labels_ #获取聚类标签
    centroids = cluster0.cluster_centers_ #获取聚类中心
    centroids = np.matrix(centroids)

    #*************************** 保存聚类结果 **********************************
    for i in range(0,num_block):
        for j in range(0,num_block):       
            bloc_im_li[i][j].set = label_pred[num_block*i+j]
    centroids = np.transpose(centroids)
    
    if(flag):
        print('All of dictionaries has been learned by pca_ISTA.')
    #*************************** 稀疏编码 **********************************
    for i in range(0,K):
        idx = np.where(label_pred == i)[0][:]
        Ck = c_Y[idx,:].T
        Omega = np.cov(Ck)
        Dic,D,V = np.linalg.svd(Omega)
        dic_li.append(Dictionary(Dic,i,centroids[:,i]))
        for id in idx:
            b = np.array(Y[:,id])
            rec = ISTA(Dic,b,tau = 1)
            row = int(id/num_block)
            column = id%num_block
            bloc_im_li[row][column].sparse_code = rec
            #print('The class '+str(i)+' has been encoded.')

    #*************************** 重建原图像 **********************************     
    N_image = np.zeros((m,m))
    for i in range(0,num_block):
        for j in range(0,num_block):
            k = bloc_im_li[i][j].set
            D = dic_li[k].dic
            n_image =  D @ bloc_im_li[i][j].sparse_code
            n_image[n_image >= 255] = 255
            n_image[n_image <= 0] = 0
            n_image = n_image.reshape((size_block,size_block),order = 'F')
            N_image[size_block*i:size_block*i+size_block,size_block*j:size_block*j+size_block] = n_image

    cur2 = datetime.datetime.now()
    if(flag):
        print('End at:  '+str(cur2))
    
    #PSNR = psnr(N_image,image)
    if(flag):
        print('Time Cost: '+str(cur2-cur1))
        #print('重建质量PSNR: '+str(PSNR))

    return N_image,bloc_im_li,dic_li

'''============================================================================
FUNCTION:ISTA_with_PCAD
FEATURE: 用已经训练号的PCA字典+ISTA编码做字典学习
INPUTS:
   image---------原始图像
   size_block----图像块大小
   K-------------聚类个数
   block_im_li---分块图像列表
   dict_li-------字典列表
OUTPUTS:
   N_image-------重建之后的图像
   bloc_im_li----图像块列表(其中包含稀疏编码)
============================================================================'''
def ISTA_with_PCAD(image,size_block,K,block_im_li,dict_li):
    m = image.shape[0]
    #分块
    num_block = int(m/size_block)#分块个数
    bloc_im_li = [[] for i in range(0,num_block)]#分块图像矩阵
    for i in range(0,num_block):
        for j in range(0,num_block):
            sub_image = image[size_block*i:size_block*i+size_block,size_block*j:size_block*j+size_block]
            block_image = Block_image(i,j,sub_image,size_block)
            bloc_im_li[i].append(block_image)
      
    cur1 = datetime.datetime.now()
    print('Begin at:  '+str(cur1))

    #*************************** KNN聚类 **********************************
    label_pred = []
    Y = np.matrix(np.zeros((size_block*size_block,num_block*num_block)))
    for i in range(0,num_block):
        for j in range(0,num_block):
            Y[:,num_block*i+j] = bloc_im_li[i][j].row_stretch
            label_pred.append(block_im_li[i][j].set)
    label_pred = np.array(label_pred)

    #*************************** 保存聚类结果 **********************************
    for i in range(0,num_block):
        for j in range(0,num_block):       
            bloc_im_li[i][j].set = block_im_li[i][j].set
    
    
    print('All of dictionaries has been learned by pca_ISTA.')
    #*************************** 稀疏编码 **********************************
    for i in range(0,K):
        idx = np.where(label_pred == i)[0][:]
        Dic = dict_li[i].dic
        for id in idx:
            b = np.array(Y[:,id])
            rec = ISTA(Dic,b,tau = 1)
            row = int(id/num_block)
            column = id%num_block
            bloc_im_li[row][column].sparse_code = rec
        #print('The class '+str(i)+' has been encoded.')

    #*************************** 重建原图像 **********************************     
    N_image = np.zeros((m,m))
    for i in range(0,num_block):
        for j in range(0,num_block):
            k = bloc_im_li[i][j].set
            D = dict_li[k].dic
            n_image =  D @ bloc_im_li[i][j].sparse_code
            n_image[n_image >= 255] = 255
            n_image[n_image <= 0] = 0
            n_image = n_image.reshape((size_block,size_block),order = 'F')
            N_image[size_block*i:size_block*i+size_block,size_block*j:size_block*j+size_block] = n_image

    cur2 = datetime.datetime.now()
    print('End at:  '+str(cur2))
    
    #PSNR = psnr(N_image,image)
    print('Time Cost: '+str(cur2-cur1))
    #print('重建质量PSNR: '+str(PSNR))

    return N_image,bloc_im_li

'''============================================================================
FUNCTION:pca_encode
FEATURE: 调用一种预处理方法
参数说明：
   flag-------------是否做字典学习
返回值：pca_ISTA,ISTA_with_PCAD
   pca_ISTA---------返回PCA字典学习函数
   ISTA_with_PCAD---返回编码函数
============================================================================'''
def pca_encode(flag):
    if(flag):
        return pca_ISTA
    else:
        return ISTA_with_PCAD

'''============================================================================
FUNCTION:ksvd
FEATURE: 用ksvd算法做字典学习
INPUTS:
   Y-------数据集合矩阵
   K-------稀疏度
   n-------字典列的个数
OUTPUTS:
   A------字典
   X------稀疏编码矩阵
   PSNR---重建的PSNR
============================================================================'''  
def ksvd(Y,A0,K,n,Acc):
    num_signal = Y.shape[1]#待分解的信号个数
    #A = overDct(m,n)
    #A = overDct(529,529)
    #A = A[0:512,0:512]
    A = A0
    IterK = 0
    #由于的DCT字典所以不必归一化
    Err = 10**7
    PSNR = 0
    X = np.matrix(np.zeros((n,num_signal)))
    
    while(IterK<Acc):
        
        IterK += 1
        
        #lsomp做稀疏表示
        for i in range(0,num_signal):
            b = Y[:,i]
            rec = lsomp(A,b,K)
            X[:,i] = rec
        
        #用KSVD做字典学习
        for j0 in range(0,n):
            Xj0 = X[j0,:]
            Xj0 = np.array(Xj0)
            Xj0.shape = [num_signal]
            InstacnceSet = np.nonzero(Xj0)
            InstacnceSet = np.array(InstacnceSet)
            InstacnceSet.shape = [InstacnceSet.shape[1]]
            InstacnceSet = list(InstacnceSet)
            if(max(np.abs(Xj0)) == 0):
                Err = Y-A*X
                Err = np.diagonal(np.transpose(Err)*Err)
                Err = list(Err)
                i = Err.index(max(Err))
                A[:,j0] = Y[:,i]
                A[:,j0] = A[:,j0]/np.linalg.norm(np.transpose(A[:,j0])*A[:,j0])
                A[:,j0] = A[:,j0]/(np.sign(A[1,j0]))
            else:
                tempX = X[:,InstacnceSet]
                tempX[j0,:] = 0
                Ej0 = Y[:,InstacnceSet]-A*tempX
                U,Delta,V = np.linalg.svd(Ej0)
                A[:,j0] = U[:,0]
                X[j0,InstacnceSet] = Delta[0]*V[0,:]
        PSNR = psnr(np.array(Y),np.array(A*X))
        curt = datetime.datetime.now()
        print('The No.'+str(IterK)+' iteration.'+'PSNR = '+str(PSNR)+'  time:'+str(curt))
    return A,X,PSNR
   

'''============================================================================
FUNCTION:mod
FEATURE: 用MOD算法做字典学习
INPUTS:
   Y-------数据集合矩阵
   K-------稀疏度
   n-------字典列的个数
OUTPUTS:
   A------字典
   X------稀疏编码矩阵
   PSNR---重建的PSNR
============================================================================'''  
def mod(Y,A0,K,n):
	#A = overDct(Y.shape[0],n)
	A = A0
	num_signal = Y.shape[1]
	IterK = 0
	X = np.matrix(np.zeros((n,num_signal)))
	PSNR = 0
	while(IterK<40):
		IterK += 1
		for i in range(num_signal):
			rec = lsomp(A,Y[:,i],K)
			X[:,i] = rec
		#用MOD
		A = Y*X.T*(X*X.T).I
		PSNR = psnr(np.array(Y),np.array(A*X))
		curt = datetime.datetime.now()
		print('The No.'+str(IterK)+' iteration.'+'PSNR = '+str(PSNR)+'  time:'+str(curt))
	return A,X,PSNR




