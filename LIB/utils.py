# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:26:57 2021

@author: 月光下的云海
"""

import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from warnings import filterwarnings
filterwarnings('ignore')


GetAbsMax = lambda x:np.abs(x).max()
GetAbsMin = lambda x:np.abs(x).min()
ave = lambda x:sum(x)/len(x)

PI = np.pi

"""===============================================================================================
Introduction: The functin to block an image into patches. The idea is quoted from W.S. Dong NCSR
(Dong W , Zhang L , Shi G . Nonlocally Centralized Sparse Representation for Image Restoration[J]. 
IEEE Transactions on Image Processing, 2013, 22(4).)
---------------------------------------------------------------------------------------------------
Function: GetPatches
Input: image,b,s,need_flatten = True,completeness = False
		image			----(numpy) inputed image
		b				----(int) block size
		s				----(int) block step
		need_flatten	----(BOOL) If it is True, the function would flatten every image patche into 
							the flattened vector. Else, the patches would be stacked straightly.
		completeness	----(BOOL) If it is True, the function would block image with stride=1 at the 
							end of image, so as to avoid miss info which out of step. Else, the function 
							would straightly droup out the out-step info.
Return: Px,ch
        Px              ----(numpy) The block image mtx.
        ch              ----(int) Channels.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def GetPatches(image,b,s,need_flatten = True,completeness = False):
    if len(image.shape) == 2:
        h,w = image.shape
        ch = 1
        N = h-b+1
        M = w-b+1
        r = np.arange(0,N,s)
        if completeness:
            r = np.hstack((r,np.arange(r[-1]+1,N)))
        c = np.arange(0,M,s)
        if completeness:
            c = np.hstack((c,np.arange(c[-1]+1,M)))
        L = r.shape[0]*c.shape[0]
        Px = np.zeros((b*b,L))
        k = 0
        for i in range(b):
            for j in range(b):
                blk = image[r+i,:]
                blk = blk[:,c+j]
                li = [blk[:,:] for k in range(ch)]
                flatten_blk = np.vstack(li).reshape((1,-1),order = 'F')
                flatten_blk = np.squeeze(flatten_blk)
                Px[k,:] = flatten_blk
                k = k+1
        if not need_flatten:
            Px = Px.reshape((b,b,L))
    elif len(image.shape) == 3:
        h,w,ch = image.shape
        N = h-b+1
        M = w-b+1
        r = np.arange(0,N,s)
        if completeness:
            r = np.hstack((r,np.arange(r[-1]+1,N)))
        c = np.arange(0,M,s)
        if completeness:
            c = np.hstack((c,np.arange(c[-1]+1,M)))
        L = r.shape[0]*c.shape[0]
        Px = np.zeros((L,b*b,3))
        k = 0
        for i in range(b):
            for j in range(b):
                blk = image[r+i,:,:]
                blk = blk[:,c+j,:]
                li = [blk[:,:,k] for k in range(ch)]
                flatten_blk = np.hstack(li).reshape((-1,1,3),order = 'F')
                #flatten_blk = np.squeeze(flatten_blk)
                Px[:,k,:] = np.squeeze(flatten_blk)
                k = k+1
        if not need_flatten:
            Px = Px.reshape((L,b,b,ch))
    return Px,ch

'''=========================================================================================
Introduction: The functin to restore the original image from patches.
---------------------------------------------------------------------------------------------------
Function: RestoreImage
Input: X,step,x_h,x_w
		X			----(numpy) inputed Patches
		s			----(int) block step
		x_h			----(int) or_image height
		x_w	        ----(int) or_image weight
Return: re_img      ----(numpy) restored image
---------------------------------------------------------------------------------------------------
Created on Thu Jan 28 16:53:04 2021
@author: 月光下的云海
========================================================================================='''
def RestoreImage(X,s,x_h,x_w):
    bS = int(X.shape[0]**0.5)
    num_blks = int(X.shape[1]**0.5)
    W = np.zeros(shape = (x_h,x_w))
    ww = np.arange(1,1+bS/s).reshape((-1,1))@np.arange(1,1+bS/s).reshape((1,-1))
    ww = np.kron(ww,np.ones((s,s)))
    W[:bS,:bS] = ww
    W[-bS:,:bS] = np.rot90(ww)
    W[:bS,-bS:] = np.flip(ww,axis = 1)
    W[-bS:,-bS:] = np.rot90(ww,2)
    W[:,bS:-bS] = np.kron(W[:,bS-1].reshape((-1,1)),np.ones((1,x_w-2*bS)))
    W[bS:-bS,:] = np.kron(W[bS-1,:].reshape((1,-1)),np.ones((x_h-2*bS,1)))
    re_img = np.zeros((x_h,x_w))
    
    for i in range(num_blks):
        for j in range(num_blks):
            re_img[s*i:s*i+bS,s*j:s*j+bS] += X[:,num_blks*i+j].reshape((bS,bS),order = 'F')
    re_img = re_img/W
    re_img = re_img.T
    return re_img

'''================== prepareImageData准备图片训练数据 =====================
FUNCTION:   prepareImageData
FEATURE:    prepareImageData准备图片训练数据
INPUTS:source_dir,ratio,subimg_h,subimg_w
       source_dir----------数据路径
       ratio---------------测试集合的比例
       subimg_h------------分割图像块高度
       subimg_w------------分割图像块宽度
OUTPUT:train_img,label_img,test_img,test_label_img
       train_img-----------训练数据（输入）
       label_img-----------训练数据（输出）
       test_img------------测试数据（输入）
       test_label_img------测试数据（输出）
========================================================================'''
def prepareImageData(
        source_dir,
        ratio,
        scale,
        subimg_h = 16,
        subimg_w = 16):
    
    for i in range(int(100*(1-ratio))):
        filename = source_dir+str(i)+'.png'
        img = Image.open(filename).convert('L')
        img1 = img.resize((512,512))
        img2 = img1.resize((512//scale,512//scale))
        img2 = DownScale(img1,(512,512),scale)
        
        np_img1 = np.asarray(img1)
        np_img1 = np_img1.reshape((512,512,1))
        img_h, img_w, _ = np_img1.shape
        np_img1 = np.lib.stride_tricks.as_strided(np_img1,
                                                  shape=(img_h // subimg_h, img_w // subimg_w, subimg_h, subimg_w),  # rows, cols
                                                  strides=np_img1.itemsize * np.array([subimg_h * img_w , subimg_w , img_w , 1 ]))
        np_img1 = np_img1.reshape(((img_h // subimg_h)*(img_w // subimg_w),subimg_h,subimg_w))
        np_img2 = np.asarray(img2)
        np_img2 = np_img2.reshape((512,512,1))
        np_img2 = np.lib.stride_tricks.as_strided(np_img2,
                                                  shape=(img_h // subimg_h, img_w // subimg_w, subimg_h, subimg_w),  # rows, cols
                                                  strides=np_img1.itemsize * np.array([subimg_h * img_w , subimg_w , img_w , 1]))
        np_img2 = np_img2.reshape(((img_h // subimg_h)*(img_w // subimg_w),subimg_h,subimg_w))
        if i == 0:
            label_img = np_img1
            train_img = np_img2
        else:
            label_img = np.vstack((label_img,np_img1))
            train_img = np.vstack((train_img,np_img2))
    train_img = train_img.reshape((train_img.shape[0],subimg_h,subimg_w,1))
    label_img = label_img.reshape((label_img.shape[0],subimg_h,subimg_w,1))
    for i in range(int(100*(1-ratio)),100):
        filename = source_dir+str(i)+'.png'
        img = Image.open(filename).convert('L')
        img1 = img.resize((512,512))
        img2 = img1.resize((512//scale,512//scale))
        img2 = DownScale(img1,(512,512),scale)
        np_img1 = np.asarray(img1)
        np_img1 = np_img1.reshape((512,512,1))
        img_h, img_w, _ = np_img1.shape
        np_img1 = np.lib.stride_tricks.as_strided(np_img1,
                                                  shape=(img_h // subimg_h, img_w // subimg_w, subimg_h, subimg_w),  # rows, cols
                                                  strides=np_img1.itemsize * np.array([subimg_h * img_w , subimg_w , img_w , 1 ]))
        np_img1 = np_img1.reshape(((img_h // subimg_h)*(img_w // subimg_w),subimg_h,subimg_w))
        np_img2 = np.asarray(img2)
        np_img2 = np_img2.reshape((512,512,1))
        np_img2 = np.lib.stride_tricks.as_strided(np_img2,
                                                  shape=(img_h // subimg_h, img_w // subimg_w, subimg_h, subimg_w),  # rows, cols
                                                  strides=np_img1.itemsize * np.array([subimg_h * img_w , subimg_w , img_w , 1]))
        np_img2 = np_img2.reshape(((img_h // subimg_h)*(img_w // subimg_w),subimg_h,subimg_w))
        if i == int(100*(1-ratio)):
            test_label_img = np_img1
            test_img = np_img2
        else:
            test_label_img = np.vstack((test_label_img,np_img1))
            test_img = np.vstack((test_img,np_img2))
    test_img = test_img.reshape((test_img.shape[0],subimg_h,subimg_w,1))
    test_label_img = test_label_img.reshape((test_label_img.shape[0],subimg_h,subimg_w,1))
    return train_img,label_img,test_img,test_label_img

'''================= prepareSparseData准备稀疏的训练数据 ===================
FUNCTION:   prepareSparseData
FEATURE:    prepareSparseData准备稀疏的训练数据
INPUTS:self,block_im_path,ratio
       self----------------自身对象
       block_im_path-------训练集合的路径
       ratio---------------测试集合的比例
OUTPUT:x_test,y_test,x_train,y_train,train_size,test_szie
       x_test--------------测试数据（输入）
       y_test--------------测试数据（输出）
       x_train-------------训练数据（输入）
       y_train-------------训练数据（输出）
       train_size----------训练集合大小
       test_szie-----------测试集合大小
========================================================================'''
def prepareSparseData(block_im_path,ratio):
    
    with np.load(block_im_path) as train_data:
        print(train_data['introduce'])
        saved_x_train = train_data['xtrain']
        saved_y_train = train_data['ytrain']
    # saved_x_train = np.load(block_im_path+'xtrain_seq.npy')
    # saved_y_train = np.load(block_im_path+'ytrain_seq.npy')

    #************* 划分训练集与测试集 ***************
    train_size = saved_x_train.shape[0]
    saved_x_train = saved_x_train.reshape((train_size,16,16,1))
    saved_y_train = saved_y_train.reshape((train_size,16,16,1))
    test_szie = int(train_size*ratio)
    x_test = saved_x_train[train_size-test_szie:train_size]
    y_test = saved_y_train[train_size-test_szie:train_size]
    x_train = saved_x_train[0:train_size-test_szie]
    y_train = saved_y_train[0:train_size-test_szie]
    train_size = x_train.shape[0]
    test_szie = x_test.shape[0]
    return x_test,y_test,x_train,y_train,train_size,test_szie


'''============================================================================
FUNCTION:overDct
FEATURE: 构建一个过完备的DCT字典
INPUTS:
   raw-------字典的行数
   column----字典的列数
OUTPUTS:
   过完备DCT字典A
Tip：raw,column必须开方为整数
============================================================================'''  
def overDct(raw,column):
    MM = raw**0.5;NN = column**0.5;
    A1 = np.matrix([i for i in range(0,int(MM))])
    A2 = np.matrix([i for i in range(0,int(NN))])
    A = ((2/MM)**0.5)*np.cos((PI/(2*MM))*(np.transpose(A1)*A2))
    A[0,:] = A[0,:]/(2**0.5)
    A = np.kron(A,A)
    return(A)

from skimage.measure import compare_psnr,compare_ssim

def psnr(img1,img2):
    return compare_psnr(img1,img2,255)

def ssim(im1,im2):
    return compare_ssim(im1,im2,multichannel=True)

'''============================================================================
FUNCTION:psnr
FEATURE: 计算两个图像间的psnr值
INPUTS:
       img1----------低分图片
       img2----------高分图像
OUTPUTS:
   psnr值
============================================================================'''  
def psnr_np(img1,img2):
    mse = np.mean((img1/1.0 - img2/1.0)**2)
    if(mse<1e-1):
        return 100
    return 10*np.log10(255.0**2/mse)

'''这个我也不知是啥'''
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
'''这个我也不知是啥'''
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

'''============================================================================
FUNCTION:ssim
FEATURE: 计算两个图像间的ssim值
INPUTS:
       img1----------低分图片
       img2----------高分图像
OUTPUTS:
   ssim值
============================================================================'''  
def ssim_np(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    return np.mean(np.mean(ssim_map))

def mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse
    
def mae(img1, img2):
    mae = np.mean( abs(img1 - img2)  )
    return mae    

'''========================= 给图像加下采样后加高斯模糊 ========================
FUNCTION:   DownSampling_n_GaussianBlur
FEATURE:    给图像加下采样后加高斯模糊
INPUTS:img,scale
       img-------------PIL图像对象(Format:Image.open('lena.bmp'))
       size_image------图像大小(Format:(512,512))
       scale-----------下采样因子(Format:2)
       gauss_radius----高斯模糊核方差(Format:1.5)
OUTPUT:img-------------下采样加模糊之后的图像对象
============================================================================='''
def DownScale_n_GaussianBlur(img,size_image,scale = 2,gauss_radius = 1.5):
    from PIL.ImageFilter import GaussianBlur
    size_im = np.array(size_image)
    img = img.resize((size_im//scale))
    img = img.resize((size_im),Image.BICUBIC)
    img = img.filter(GaussianBlur(radius = gauss_radius))
    return img

'''============================= 给图像加高斯模糊 ==============================
FUNCTION:   GaussianBlur
FEATURE:    给图像加高斯模糊
INPUTS:img,size_image,gauss_radius
       img-------------PIL图像对象(Format:Image.open('lena.bmp'))
       size_image------图像大小(Format:(512,512))
       scale-----------下采样因子(Format:2)
       gauss_radius----高斯模糊核方差(Format:1.5)
OUTPUT:img-------------下采样加模糊之后的图像对象
============================================================================='''
def GaussianBlur(img,size_image,gauss_radius = 1.5):
    return DownScale_n_GaussianBlur(img,size_image,scale = 1,gauss_radius = 1.5)

'''================================ 给图像下采样 ===============================
FUNCTION:   DownSampling
FEATURE:    给图像下采样
INPUTS:img,size_image,scale
       img---------PIL图像对象(Format:Image.open('lena.bmp'))
       size_image--图像大小(Format:(512,512))
       scale-------下采样因子(Format:2)
OUTPUT:img---------下采样加模糊之后的图像对象
============================================================================='''
def DownScale(img,size_image,scale = 2,gauss_radius = 1.5):
    size_im = np.array(size_image)
    img = img.resize((size_im//scale))
    img = img.resize((size_im))
    return img

'''============================= 给图像加Gauss噪声 =============================
FUNCTION:   AddGaussNoise
FEATURE:    给图像加Gauss噪声
INPUTS:img,mu,std
       img---------PIL图像对象(Format:Image.open('lena.bmp'))
       mu----------噪声水平(Format:10)
       scale-------噪声方差(Format:1)
OUTPUT:img---------下采样加模糊之后的图像对象
============================================================================='''
def AddGaussNoise(img,mu,std = 1):
    img = np.array(img)
    noise = np.random.normal(0, std, img.shape)
    out = img + mu*noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return Image.fromarray(out)

'''============================= 给图像加下采样后加高斯模糊加Gauss噪声 =============================
FUNCTION:   AddGaussNoise
FEATURE:    给图像加Gauss噪声
INPUTS:img,mu,std
       img---------PIL图像对象(Format:Image.open('lena.bmp'))
       mu----------噪声水平(Format:10)
       scale-------噪声方差(Format:1)
OUTPUT:img---------下采样加模糊之后的图像对象
============================================================================='''
def DownScale_n_GaussianBlur_n_AddGaussNoise(img,size_image,scale = 2,gauss_radius = 1.5):
    img = DownScale_n_GaussianBlur(img,size_image,scale,gauss_radius)
    img = AddGaussNoise(img,mu = 5,std = 1)
    return img

