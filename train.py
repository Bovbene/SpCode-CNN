# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:26:35 2021

@author: 月光下的云海
"""
from LIB.utils import prepareImageData,prepareSparseData
import tensorflow as tf
from warnings import filterwarnings
filterwarnings('ignore')
from json import load

def train_srcnn(config):
    from MODELS.SRCNN import SRCNN
    model = SRCNN(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/SRCNNx{}/'.format(config["up_scale"]))
    train_img,label_img,test_img,test_label_img = prepareImageData(
        './DATABASE/DIV2K100/',0.2,scale = config["up_scale"],subimg_h = 32,subimg_w = 32)
    print('\n\nSRModel (' + 'SRCNN' +' x '+str(config["up_scale"])+ ') Trainning ... ...')
    model.train(train_img,
                label_img,
                test_img,
                test_label_img,
                Epoch = config["epoch"],
                iter_view = config["iter_view"])
    print('----<The training process of SRCNN has been complished.>----')

def train_vdsr(config):
    from MODELS.VDSR import VDSR
    
    model = VDSR(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/VDSRx{}/'.format(config["up_scale"]))
    train_img,label_img,test_img,test_label_img = prepareImageData(
        './DATABASE/DIV2K/',0.2,scale = config["up_scale"],subimg_h = 32,subimg_w = 32)
    print('\n\nSRModel (' + 'VDSR' +' x '+str(config["up_scale"])+ ') Trainning ... ...')
    model.train(train_img,
                label_img,
                test_img,
                test_label_img,
                Epoch = config["epoch"],
                iter_view = config["iter_view"])
    print('----<The training process of VDSR has been complished.>----')

def train_SpCodeSRCNN(config):
    from MODELS.SRCNN import SRCNN
    model = SRCNN(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/SpCode-SRCNNx{}/'.format(config["up_scale"]))          
    x_test,y_test,x_train,y_train,train_size,test_size = prepareSparseData(
        './DATABASE/TrainDatax{}.npz'.format(config["up_scale"]),0.2)
    print('\n\SpCode- ( ' + 'SRCNN' +' x '+str(config["up_scale"])+ ' ) Trainning ... ...')
    model.train(x_train,
                y_train,
                x_test,
                y_test,
                Epoch = config["epoch"],
                iter_view = config["iter_view"])
    print('----<The training process of SpCode-SRCNN has been complished.>----')

def train_SpCodeVDSR(config):
    from MODELS.VDSR import VDSR
    model = VDSR(config = config,sess = tf.Session(),save_path = './TRAINED_MODEL/SpCode-VDSRx{}/'.format(config["up_scale"]))        
    x_test,y_test,x_train,y_train,train_size,test_size = prepareSparseData(
        './DATABASE/TrainDatax{}.npz'.format(config["up_scale"]),0.2)
    print('\n\SpCode- ( ' + 'VDSR' +' x '+str(config["up_scale"])+ ' ) Trainning ... ...')
    model.train(x_train,
                y_train,
                x_test,
                y_test,
                Epoch = config["epoch"],
                iter_view = config["iter_view"])

    print('----<The training process of SpCode-VDSR has been complished.>----')

def main(args):
    with open('./config.json','r') as f:
        config = load(f)
        f.close()
    if args.which_model == 'SRCNN':
        train_srcnn(config["srcnn_config"])
    elif args.which_model == 'VDSR':
        train_vdsr(config["vdsr_config"])
    elif args.which_model == 'SpCode-SRCNN':
        train_SpCodeSRCNN(config["spcode_srcnn"])
    elif args.which_model == 'SpCode-VDSR':
        train_SpCodeVDSR( config["spcode_vdsr"])

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description = "Main Function of That Use PCA n' ISTA Encode All Images")
    parser.add_argument('-WM','--which_model',
                        choices = ['SRCNN','VDSR','SpCode-SRCNN','SpCode-VDSR'],
                        help="Choose one type of model to train or test.")
    args = parser.parse_args()
    main(args)