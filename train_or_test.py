# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:14:46 2021

@author: 月光下的云海
"""

from LIB.utils import prepareImageData,DownScale,psnr,ssim,prepareSparseData
import tensorflow as tf
from PIL import Image
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

def train_test_srcnn(is_train,scale = 4):
    from MODELS.SRCNN import SRCNN
    model = SRCNN(scale = scale,sess = tf.Session(),save_path = './TRAINED_MODEL/SRCNNx{}/'.format(scale))
    if is_train:
        train_img,label_img,test_img,test_label_img = prepareImageData(
            './DATABASE/DIV2K100/',0.2,scale = 4,subimg_h = 32,subimg_w = 32)
        print('\n\nSRModel (' + 'SRCNN' +' x '+str(scale)+ ') Trainning ... ...')
        model.train(train_img,
                    label_img,
                    test_img,
                    test_label_img,
                    Epoch = int(5e3),
                    iter_view = 500)
    else:
        or_image = Image.open('./DATABASE/Set5/butterfly_GT.bmp').convert("YCbCr")
        lr_image = DownScale(or_image,or_image.size,scale = 4)
        lr_image = np.array(lr_image)
        sr_image = model.test(image = lr_image)
        sr_image.show()
        print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
              .format(psnr(np.array(or_image),np.array(sr_image)),ssim(np.array(or_image),np.array(sr_image))))

def train_test_vdsr(is_train,scale = 4):
    from MODELS.VDSR import VDSR
    model = VDSR(scale = scale,sess = tf.Session(),save_path = './TRAINED_MODEL/VDSRx{}/'.format(scale))
    if is_train:
        train_img,label_img,test_img,test_label_img = prepareImageData(
            './DATABASE/DIV2K/',0.2,scale = 4,subimg_h = 32,subimg_w = 32)
        print('\n\nSRModel (' + 'VDSR' +' x '+str(scale)+ ') Trainning ... ...')
        model.train(train_img,
                    label_img,
                    test_img,
                    test_label_img,
                    Epoch = int(5e4),
                    iter_view = 500)
    else:
        or_image = Image.open('./DATABASE/Set5/butterfly_GT.bmp').convert("YCbCr")
        lr_image = DownScale(or_image,or_image.size,scale = 4)
        lr_image = np.array(lr_image)
        sr_image = model.test(image = lr_image)
        sr_image.show()
        print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
              .format(psnr(np.array(or_image),np.array(sr_image)),ssim(np.array(or_image),np.array(sr_image))))

def train_test_SpCodeSRCNN(is_train,scale):
    from MODELS.SRCNN import SRCNN
    model = SRCNN(scale = scale,sess = tf.Session(),save_path = './TRAINED_MODEL/SpCode-SRCNNx{}/'.format(scale))
    if is_train:            
        x_test,y_test,x_train,y_train,train_size,test_size = prepareSparseData(
            './DATABASE/TrainDatax4.npz',0.2)
        print('\n\SpCode- ( ' + 'SRCNN' +' x '+str(scale)+ ' ) Trainning ... ...')
        model.train(x_train,
                    y_train,
                    x_test,
                    y_test,
                    Epoch = int(5e4),
                    iter_view = 500)
    else:
        filename = './DATABASE/Set5/woman_GT.bmp'
        hr_image = np.asarray(Image.open(filename).convert("YCbCr"))
        sr_image = model.SpCodeSRCNNSR(filename)
        print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
              .format(psnr(hr_image,sr_image),ssim(hr_image,sr_image)))

def train_test_SpCodeVDSR(is_train,scale):
    from MODELS.VDSR import VDSR
    model = VDSR(scale = scale,sess = tf.Session(),save_path = './TRAINED_MODEL/SpCode-VDSRx{}/'.format(scale))
    if is_train:            
        x_test,y_test,x_train,y_train,train_size,test_size = prepareSparseData(
            './DATABASE/TrainDatax4.npz',0.2)
        print('\n\SpCode- ( ' + 'VDSR' +' x '+str(scale)+ ' ) Trainning ... ...')
        model.train(x_train,
                    y_train,
                    x_test,
                    y_test,
                    Epoch = int(5e4),
                    iter_view = 500)
    else:
        filename = './DATABASE/Set5/woman_GT.bmp'
        hr_image = np.asarray(Image.open(filename).convert("YCbCr"))
        sr_image = model.SpCodeVDSRSR(filename)
        print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
              .format(psnr(hr_image,sr_image),ssim(hr_image,sr_image)))

def main(args):
    is_train = args.is_train
    scale = args.up_scale
    if args.which_model == 'SRCNN':
        train_test_srcnn(is_train,scale)
    elif args.which_model == 'VDSR':
        train_test_vdsr(is_train,scale)
    elif args.which_model == 'SpCode-SRCNN':
        train_test_SpCodeSRCNN(is_train, scale)
    elif args.which_model == 'SpCode-VDSR':
        train_test_SpCodeVDSR(is_train, scale)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description = "Main Function of That Use PCA n' ISTA Encode All Images")
    parser.add_argument('-WM','--which_model',
                        choices = ['SRCNN','VDSR','SpCode-SRCNN','SpCode-VDSR'],
                        help="Choose one type of model to train or test.")
    parser.add_argument('-USC','--up_scale',type = int,
                        default = 4,
                        help="input factor of up scale.")
    parser.add_argument('-IT','--is_train',type = bool,
                        default = False,
                        help="True for training. False for testing")
    args = parser.parse_args()
    main(args)
    
    