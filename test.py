# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:34:45 2021

@author: 月光下的云海
"""

from LIB.utils import (DownScale,
                       psnr,
                       ssim,
                       DownScale_n_GaussianBlur,
                       AddGaussNoise,
                       GaussianBlur,
                       DownScale_n_GaussianBlur_n_AddGaussNoise)
import tensorflow as tf
from PIL import Image
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from json import load


def test_srcnn(scale,fn,downscale_fn,config):
    from MODELS.SRCNN import SRCNN
    model = SRCNN(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/SRCNNx{}/'.format(scale))
    model.scale = scale
    or_image = Image.open(fn).convert("YCbCr")
    lr_image = downscale_fn(or_image,or_image.size,scale = scale)
    lr_image = np.array(lr_image)
    sr_image = model.test(image = lr_image)
    sr_image.show()
    print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
          .format(psnr(np.array(or_image),np.array(sr_image)),ssim(np.array(or_image),np.array(sr_image))))

def test_vdsr(scale,fn,downscale_fn,config):
    from MODELS.VDSR import VDSR
    model = VDSR(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/VDSRx{}/'.format(scale))
    model.scale = scale
    or_image = Image.open(fn).convert("YCbCr")
    lr_image = downscale_fn(or_image,or_image.size,scale = scale)
    lr_image = np.array(lr_image)
    sr_image = model.test(image = lr_image)
    sr_image.show()
    print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
          .format(psnr(np.array(or_image),np.array(sr_image)),ssim(np.array(or_image),np.array(sr_image))))

def test_SpCodeSRCNN(scale,fn,downscale_fn,config):
    from MODELS.SRCNN import SRCNN
    model = SRCNN(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/SpCode-SRCNNx{}/'.format(scale))
    model.scale = scale
    hr_image = np.asarray(Image.open(fn).convert("YCbCr"))
    sr_image = model.SpCodeSRCNNSR(fn,downscale_fn)
    print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
          .format(psnr(hr_image,sr_image),ssim(hr_image,sr_image)))

def test_SpCodeVDSR(scale,fn,downscale_fn,config):
    from MODELS.VDSR import VDSR
    model = VDSR(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/SpCode-VDSRx{}/'.format(scale))
    model.scale = scale
    hr_image = np.asarray(Image.open(fn).convert("YCbCr"))
    sr_image = model.SpCodeVDSRSR(fn,downscale_fn)
    print('----<The PSNR, SSIM between HR image and SR image is: PSNR:{:.5f}, SSIM:{:.5f}>----'
          .format(psnr(hr_image,sr_image),ssim(hr_image,sr_image)))

def main(args):
    with open('./config.json','r') as f:
        config = load(f)
        f.close()
    if config["downscale_fn"] == config["downscale_fn_option"][0]:
        downscale_fn = DownScale
    elif config["downscale_fn"] == config["downscale_fn_option"][1]:
        downscale_fn = DownScale_n_GaussianBlur
    elif config["downscale_fn"] == config["downscale_fn_option"][2]:
        downscale_fn = GaussianBlur
    elif config["downscale_fn"] == config["downscale_fn_option"][3]:
        downscale_fn = AddGaussNoise
    elif config["downscale_fn"] == config["downscale_fn_option"][4]:
        downscale_fn = DownScale_n_GaussianBlur_n_AddGaussNoise 
    if args.which_model == 'SRCNN':
        test_srcnn(args.up_scale,args.file_name,downscale_fn,config)
    elif args.which_model == 'VDSR':
        test_vdsr(args.up_scale,args.file_name,downscale_fn,config)
    elif args.which_model == 'SpCode-SRCNN':
        test_SpCodeSRCNN(args.up_scale,args.file_name,downscale_fn,config)
    elif args.which_model == 'SpCode-VDSR':
        test_SpCodeVDSR(args.up_scale,args.file_name,downscale_fn,config)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description = "Main Function of That Use PCA n' ISTA Encode All Images")
    parser.add_argument('-WM','--which_model',default = 'SpCode-VDSR',
                        choices = ['SRCNN','VDSR','SpCode-SRCNN','SpCode-VDSR'],
                        help="Choose one type of model to train or test.")
    parser.add_argument('-USC','--up_scale',type = int,
                        default = 4,
                        help="input factor of up scale.")
    parser.add_argument('-FN', '--file_name',type = str,default = './DATABASE/Set5/woman_GT.bmp',
                        help = "The image path to test.")
    args = parser.parse_args()
    
    main(args)
    
    