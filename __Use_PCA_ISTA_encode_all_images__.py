# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:45:11 2020

@author: 月光下的云海
"""

import numpy as np
from PIL import Image
from DLL.Valuation import psnr
from DLL.DictionaryLearning import pca_encode
from DLL.DownSample import *


def pca_learning_on_all_image(
        size_block          = 16,                                    #图像块大小
        K                   = 150,                                   #聚类个数
        work_dir            = r'F:\\Project\\xTrainData\\',             #跟目录
        target_dir          = 'F:\\Project\\Saprse_Train_Data\\',   #目标文件
        dic_target_dir      = 'F:\\Project\\Dic_TrainData\\',       #目标文件
        size_image          = 512,                                   #图片大小
        num_image           = 10,                                    #图片个数10**2
        scale               = 2,                                     #下采样大小
        filename_of_dict    = 'LR_Dict',                             #要保存的字典名称
        filename_of_code    = 'LR_SparseCode',                       #要保存的稀疏编码名称
        flag                = True,                                 #是否进行字典学习
        hr_dict_li          = None,                                  #只有不使用字典学习的时候才传入参数
        hr_block_im_li      = None,                                   #只有不使用字典学习的时候才传入参数
        downsample_fn       = DownScale,
        gauss_radius        = 1.5
        ):
    #************************* 图像合并 **********************************
    merge_image = np.zeros((size_image*num_image,size_image*num_image))
    merge_image_cb = np.zeros((size_image*num_image,size_image*num_image))
    merge_image_cr = np.zeros((size_image*num_image,size_image*num_image))
    
    if not flag:
        downsample_fn = DownScale
    
    for i in range(0,num_image):
        for j in range(0,num_image):
            name = 10*i+j
            file_path = work_dir+str(name)+'.png'
            image0 = Image.open(file_path)
            image0 = downsample_fn(image0,(size_image,size_image),
                                   scale = scale,gauss_radius = gauss_radius)
            #image0 = AddGaussNoise(image0,mu = 5**0.5)
            image0 = image0.convert("YCbCr")
            image = np.array(image0)
            
            merge_image_cb[size_image*i:size_image*i+size_image,
                        size_image*j:size_image*j+size_image] = image[:,:,1]
            merge_image_cr[size_image*i:size_image*i+size_image,
                        size_image*j:size_image*j+size_image] = image[:,:,2]
            
            merge_image[size_image*i:size_image*i+size_image,
                        size_image*j:size_image*j+size_image] = image[:,:,0]
    or_image = np.stack((merge_image,merge_image_cb,merge_image_cr),axis = 2)
    Image.fromarray(np.uint8(or_image),"YCbCr").show()

    #************************* 图像编码 **********************************
    print('The encoding process begin!')
    DicLearning = pca_encode(flag)
    if(flag):
        N_image,bloc_im_li,dic_li = DicLearning(merge_image,size_block,K)
        dic_numpy = np.array(dic_li)
        bloc_im_numpy = np.array(bloc_im_li)
        np.save(dic_target_dir+'\\'+filename_of_dict,dic_numpy)
        np.save(target_dir+'\\'+filename_of_code,bloc_im_numpy)
    else:
        N_image,bloc_im_li = DicLearning(merge_image,size_block,K,hr_block_im_li,hr_dict_li)
        dic_numpy = None
        bloc_im_numpy = np.array(bloc_im_li)
        np.save(target_dir+'\\'+filename_of_code,bloc_im_numpy)
    
    new_image = np.stack((N_image,merge_image_cb,merge_image_cr),axis = 2)
    PSNR = psnr(new_image,or_image)
    print('重建质量PSNR: '+str(PSNR))
    Image.fromarray(np.uint8(new_image),"YCbCr").show()
    return dic_numpy,bloc_im_numpy
  

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    parser = ArgumentParser(description = "Main Function of That Use PCA n' ISTA Encode All Images")
    parser.add_argument('-NI','--num_image',
                        default = 10,
                        help="input the numbers of images--10 means 10**2 images be choiced.")
    parser.add_argument('-USC','--up_scale',
                        default = 2,
                        help="input factor of up scale.")
    parser.add_argument('-TD','--target_dir',
                        default = str('F:/Project/Project8/Saprse_Train_Data/'),
                        help="input dir to save result.")
    parser.add_argument('-DSF','--down_sample_fn',
                        choices = ['DownScale','DownScale_n_GaussianBlur','GaussianBlur','AddGaussNoise','DownScale_n_GaussianBlur_n_AddGaussNoise'],
                        help = "the method of downsampling orginal image.")
    parser.add_argument('-GR','--gauss_radius',
                        default = 1.5,
                        help = "the radius of gaussian blur kernel.")
    args = parser.parse_args()
    
    NUM_IMAGE       = int(args.num_image)
    UP_SCALE        = int(args.up_scale)
    TARGET_DIR      = args.target_dir
    
    if args.down_sample_fn == 'DownScale':
        downsample_fn = DownScale
    elif args.down_sample_fn == 'DownScale_n_GaussianBlur':
        downsample_fn = DownScale_n_GaussianBlur
    elif args.down_sample_fn == 'GaussianBlur':
        downsample_fn = GaussianBlur
    elif args.down_sample_fn == 'AddGaussNoise':
        downsample_fn = AddGaussNoise
    elif args.down_sample_fn == 'DownScale_n_GaussianBlur_n_AddGaussNoise':
        downsample_fn = DownScale_n_GaussianBlur_n_AddGaussNoise
    else:
        downsample_fn = DownScale
    
    from os import path,makedirs
    if not path.exists(TARGET_DIR):
        makedirs(TARGET_DIR)
        
    #NUM_IMAGE           = 10
    #UP_SCALE            = 2
    #target_dir          = 'F:\\Project\\Project8\\Saprse_Train_Data\\'   #目标文件
    lr_dic_li,lr_block_li = pca_learning_on_all_image(
            scale               = UP_SCALE,                         #下采样大小
            filename_of_dict    = 'LR_Dictx'+str(UP_SCALE),         #要保存的字典名称
            num_image           = NUM_IMAGE,
            filename_of_code    = 'LR_SparseCodex'+str(UP_SCALE),   #要保存的稀疏编码名称
            downsample_fn       = downsample_fn,
            gauss_radius        = float(args.gauss_radius),
            target_dir          = TARGET_DIR,
            dic_target_dir      = TARGET_DIR,
            work_dir            = 'F:\\Project\\xTrainData\\'
            )
    hr_dic_li,hr_block_li = pca_learning_on_all_image(
            scale               = 1,                     #下采样大小
            filename_of_dict    = 'Dict',                #要保存的字典名称
            filename_of_code    = 'SparseCode',          #要保存的稀疏编码名称
            num_image           = NUM_IMAGE,
            flag                = False,
            hr_dict_li          = lr_dic_li,
            hr_block_im_li      = lr_block_li,
            downsample_fn       = downsample_fn,
            gauss_radius        = float(args.gauss_radius),
            target_dir          = TARGET_DIR,
            dic_target_dir      = TARGET_DIR,
            work_dir            = 'F:\\Project\\yTrainData\\'
            )
    
    
    ytrain_seq = np.zeros((0,16,16))
    xtrain_seq = np.zeros((0,16,16))
    label_li = []
    for row in range(512*NUM_IMAGE//16):
        for col in range(512*NUM_IMAGE//16):
            hr_sparse = hr_block_li[row][col].sparse_code
            hr_sparse = hr_sparse.reshape((16,16),order = 'F')
            hr_sparse = hr_sparse.reshape((1,16,16))
            ytrain_seq = np.vstack((ytrain_seq,hr_sparse))
            lr_sparse = lr_block_li[row][col].sparse_code
            lr_sparse = lr_sparse.reshape((16,16),order = 'F')
            lr_sparse = lr_sparse.reshape((1,16,16))
            xtrain_seq = np.vstack((xtrain_seq,lr_sparse))
            label_li.append(hr_block_li[row][col].set)
        print('The row_{} sparse image has been encoded.'.format(row))
    np.save(TARGET_DIR+r'xtrain_seq',xtrain_seq)
    np.save(TARGET_DIR+r'ytrain_seq',ytrain_seq)
    np.save(TARGET_DIR+r'label_li',np.array(label_li))













