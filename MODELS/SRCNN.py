# -*- coding: utf-8 -*-
"""===============================================================================================
The Python code of SRCNN to realize SR. The idea is cited from "Dong C , Loy C C , He K , et al. 
Image Super-Resolution Using Deep Convolutional Networks[J]. IEEE Trans Pattern Anal Mach Intell, 
2016, 38(2):295-307."
---------------------------------------------------------------------------------------------------
Class: SRCNN
Param: 	config,sess,save_path
---------------------------------------------------------------------------------------------------
Tip: None
---------------------------------------------------------------------------------------------------
Created on Thu Mar 25 20:14:42 2021
@author: 西电博巍(Bowei Wang, QQ: 月光下的云海)
Version: Ultimate
==============================================================================================="""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from .BaseModel import BaseModel
import numpy as np
from LIB.utils import ave
from time import time,strftime,localtime
from PIL import Image
import cv2 as cv
from LIB.DictionaryLearning import pca_ISTA

class SRCNN(BaseModel):
    
    def __init__(self,config,sess,save_path):
        self.name = config["name"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.scale = config["up_scale"]
        self.sess = sess
        if save_path is None:
            raise AttributeError("U mustn input the saved path!")
        else:self.save_path = save_path#'./TRAINED_MODEL/SRCNNx{}/'.format(scale)

        
    def srcnn(self,x,scope = 'SRCNN',reuse = None):
        with tf.variable_scope(scope,reuse = reuse):
            with slim.arg_scope([slim.conv2d],activation_fn = super().lrelu):
                y = slim.conv2d(x,64,3,stride = 1,padding = 'SAME',scope = 'conv1')
                y = slim.conv2d(y,32,1,stride = 1,padding = 'SAME',scope = 'conv2')
                y = slim.conv2d(y,1,5,stride = 1,padding = 'SAME',scope = 'conv3',activation_fn = None)
                return y
    
    def train(self,
              x_train,
              y_train,
              x_test,
              y_test,
              Epoch = int(5e3),
              iter_view = 500):
        train_size,test_size,blk_size = x_train.shape[0],x_test.shape[0],x_train.shape[1]
        batch_size = self.batch_size
        tf.reset_default_graph()
        x_train_ph = tf.placeholder(shape = (None,blk_size,blk_size,1),dtype = tf.float32)
        y_train_ph = tf.placeholder(shape = (None,blk_size,blk_size,1),dtype = tf.float32)
        x_test_ph = tf.placeholder(shape = (None,blk_size,blk_size,1),dtype = tf.float32)
        y_test_ph = tf.placeholder(shape = (None,blk_size,blk_size,1),dtype = tf.float32)
        output = self.srcnn(x_train_ph)
        test_output = self.srcnn(x_test_ph,reuse = True)
        train_op,loss = super().grenerate_train(y_train_ph,output)
        psnr = tf.image.psnr(test_output, y_test_ph, max_val=255)
        ssim = tf.image.ssim(test_output, y_test_ph, max_val=255)
        saved_dir = self.save_path+'model.ckpt'
        Threshold = -float('Inf')
        self.sess = tf.Session()
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print('='*74)
        for e in range(Epoch):
            batch1 = np.random.randint(0,train_size,size = batch_size)
            t_x_train = x_train[batch1]
            t_y_train = y_train[batch1]
            batch2 = np.random.randint(0,test_size,size = batch_size)
            t_x_test = x_test[batch2]
            t_y_test = y_test[batch2]
            self.sess.run(train_op,feed_dict = {x_train_ph:t_x_train,y_train_ph:t_y_train})
            if e % iter_view == iter_view - 1:
                train_loss = self.sess.run(loss,feed_dict = {x_train_ph:t_x_train,y_train_ph:t_y_train})
                Ipsnr,Issim = self.sess.run([psnr,ssim],feed_dict = {x_test_ph:t_x_test,y_test_ph:t_y_test})
                print('[Epoch:{}],train_loss:{:.5f},test_psnr:{:.5f},test_ssim:{:.5f}'.format(e+1,train_loss,ave(Ipsnr),ave(Issim)))
                if ave(Ipsnr) > Threshold:
                    print('Saving the better model......')
                    saver.save(sess = self.sess,save_path = saved_dir)
                    Threshold = ave(Ipsnr)
                    ThSSIM = ave(Issim)
        print('='*30 + 'Train Done !!!' + '='*30)
        print('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}),(SSIM:{:.4f}).'.format(Threshold, ThSSIM))
        print(x_test_ph)
        print(test_output)
        print('='*30 + 'Train Done !!!' + '='*30)
        self.sess.close()
    
    def test(self,image = None,image_path = None):
        if (image is None) and (image_path is None):
            raise AttributeError("U must input an image path or an image mtx.")
        elif (image is None) and (image_path is not None):
            lr_image = Image.open(image_path)
            lr_image = lr_image.convert("YCbCr")
            lr_image = np.array(lr_image)
        elif (image is not None) and (image_path is None):
            lr_image = image
        else:
            raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
        Y,Cb,Cr = lr_image[:,:,0],lr_image[:,:,1],lr_image[:,:,2]
        h,w = Y.shape[0],Y.shape[1]
        lr_image = Y.reshape((1,h,w,-1))
        input_ph = tf.placeholder(shape = lr_image.shape,dtype = tf.float32,name = 'TestInputPh')
        output = self.srcnn(input_ph,reuse = tf.AUTO_REUSE)
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, self.save_path+'model.ckpt')
        t = time()
        result = self.sess.run(output,feed_dict = {input_ph:lr_image})
        result = result.reshape((h,w))
        print("Time Elapsed:",time()-t)
        result = np.stack((result,Cb,Cr),axis = 2)
        result = Image.fromarray(np.uint8(result),"YCbCr")
        self.sess.close()
        return result
    
    def SpCodeSRCNNSR(self,image_path = None,downscale_fn = None):
        nimage = Image.open(image_path)
        w,h = nimage.size
        nimage = downscale_fn(nimage,nimage.size,scale = self.scale)
        # nimage = nimage.resize((w//self.scale,h//self.scale))
        # nimage = nimage.resize((w,h))
        
        nimage = nimage.convert("YCbCr")
        nimage = np.array(nimage)
        Y,Cb,Cr = nimage[:,:,0],nimage[:,:,1],nimage[:,:,2]
        #Image.fromarray(Y).show()
        ph,pw = 512-h,512-w
        pY = cv.copyMakeBorder(Y,0,ph,0,pw,cv.BORDER_REFLECT)
        
        _,bloc_im_li,dic_li = pca_ISTA(pY,16,1,flag = False)
        x_seq,label_li = [],[]
        for row in range(32):
            for col in range(32):
                hr_sparse = bloc_im_li[row][col].sparse_code
                hr_sparse = hr_sparse.reshape((16,16),order = 'F')
                hr_sparse = hr_sparse.reshape((1,16,16))
                x_seq.append(hr_sparse)
                label_li.append(bloc_im_li[row][col].set)
        x_seq = np.vstack(x_seq)
        x_seq = x_seq.reshape(x_seq.shape+(1,))
        input_ph = tf.placeholder(shape = x_seq.shape,dtype = tf.float32,name = 'TestInputPh')
        output = self.srcnn(input_ph,reuse = tf.AUTO_REUSE)
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, self.save_path+'model.ckpt')
        t = time()
        outputs = self.sess.run(output,feed_dict = {input_ph:x_seq})
        print("Time Elapsed:",time()-t)
        self.sess.close()
        new_image = 255*np.ones((512,512))
        for i in range(32):
            for j in range(32):
                idx = 32*i+j
                sparse_code = np.squeeze(outputs[idx])
                sparse_code = sparse_code.reshape((16*16,1),order = 'F')
                label = label_li[idx]
                Dict = dic_li[label].dic
                block_image = Dict @ sparse_code
                block_image[block_image>255] = 255
                block_image[block_image<0] = 0
                block_image = block_image.reshape((16,16),order = 'F')
                new_image[16*i:16*(i+1),16*j:16*(j+1)] = block_image
        new_image = new_image[:h,:w]
        new_image = np.stack((new_image,Cb,Cr),axis = 2)
        return new_image

