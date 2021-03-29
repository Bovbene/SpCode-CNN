# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:53:12 2020

@author: 月光下的云海
"""
"""============================================================================
-------------------------------------------------------------------------------
FILENAME:CreateModel.py
FEATURE:构建三种不同的RNN网络结构，并返回结果
-------------------------------------------------------------------------------
Created on Fri June 26 10:46:06 2020
Version:Project2.0; Python version:3.6.2
@author: 月光下的云海
-------------------------------------------------------------------------------
============================================================================"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import DLL.DownSample as DS
from PIL import Image

class CreateModel():
    
    '''=========================== __init__初始化函数 ==========================
    FUNCTION:   __init__
    FEATURE:    网络模型初始化
    INPUTS:self,model,deep,size,stride,lr,batch_size
           self----------------自身对象
           model---------------模型名称
           deep----------------每个卷积层的深度
           size----------------每个卷积层的卷积核大小
           stride--------------每个卷积层的步长
           lr------------------学习率
           batch_size----------批大小
    OUTPUT:无
    ========================================================================'''
    def __init__(self,
                 model = 'SRCNN',
                 deep = [64,32,1],
                 size = [3,1,5],
                 stride = [1,1,1],
                 lr = 1e-3,
                 batch_size = 128,
                 zoom = 2
                 ):
        self.model = model
        self.deep_li = deep
        self.size_li = size
        self.stride_li = stride
        self.lr = lr
        self.batch_size = batch_size
        self.zoom = zoom
    
    '''=========================== SRCNN网络结构 ===============================
    FUNCTION:   SRCNN
    FEATURE:    SRCNN网络结构
    INPUTS:self,inputs,scope,reuse
           self----------------自身对象
           inputs--------------网络输入
           scope---------------网络名称
           reuse---------------网络是否重用
    OUTPUT:net-----------------网络返回值
    ========================================================================'''
    def SRCNN(self,inputs,scope = 'SRCNN',reuse = None):
        with tf.variable_scope(scope,reuse = reuse):
            with slim.arg_scope([slim.conv2d],activation_fn = None):
                net = slim.conv2d(inputs,
                                  self.deep_li[0],
                                  self.size_li[0],
                                  stride = self.stride_li[0],
                                  padding = 'SAME',
                                  scope = 'conv1')
                net = tf.nn.leaky_relu(net,alpha = 0.6,name = 'act1')
                net = slim.conv2d(net,
                                  self.deep_li[1],
                                  self.size_li[1],
                                  stride = self.stride_li[1],
                                  padding = 'SAME',
                                  scope = 'conv2')
                net = tf.nn.leaky_relu(net,alpha = 0.6,name = 'act1')
                net = slim.conv2d(net,
                                  self.deep_li[2],
                                  self.size_li[2],
                                  stride = self.stride_li[2],
                                  padding = 'SAME',
                                  scope = 'conv3')
                return net

    '''========================================================================
    FUNCTION:   prelu
    FEATURE:    prelu激活函数
    INPUTS:self,x,i
           self----------------自身对象
           x-------------------激活函数输入
           i-------------------层数
    OUTPUT:激活函数返回值
    ========================================================================'''
    def prelu(self,x,i):
        alphas = tf.get_variable('alpha{}'.format(i),x.get_shape()[-1],initializer=tf.constant_initializer(0.0),dtype=tf.float64)
        pos = tf.nn.relu(x)
        neg = alphas * (x-abs(x))*0.5
        return pos+neg
        
    '''=========================== FSRCNN网络结构 ==============================
    FUNCTION:   FSRCNN
    FEATURE:    FSRCNN网络结构
    INPUTS:self,inputs,scope,reuse
           self----------------自身对象
           inputs--------------网络输入
           scope---------------网络名称
           reuse---------------网络是否重用
    OUTPUT:net-----------------网络返回值
    ========================================================================'''
    def FSRCNN(self,inputs,scope = 'FSRCNN',reuse = None):
        with tf.variable_scope(scope,reuse = reuse):
            with slim.arg_scope([slim.conv2d],activation_fn = None):
                net = slim.conv2d(inputs,56,5,stride = 1,padding = 'SAME',scope = 'conv1')
                net = self.prelu(net,1)
                net = slim.conv2d(net,12,1,stride = 1,padding = 'SAME',scope = 'conv2')
                net = self.prelu(net,2)
                for i in range(3,7):
                    net = slim.conv2d(net,12,3,stride = 1,padding = 'SAME',scope = 'conv{}'.format(i))
                    net = self.prelu(net,i)
                net = slim.conv2d(net,56,1,stride = 1,padding = 'SAME',scope = 'conv8')
                net = self.prelu(net,8)
                net = slim.conv2d_transpose(net,1,9,stride = 1,padding = 'SAME',scope = 'deconv')
                return net

    '''============================= VDSR网络结构 ==============================
    FUNCTION:   VDSR
    FEATURE:    VDSR网络结构
    INPUTS:self,inputs,depth,scope,reuse
           self----------------自身对象
           inputs--------------网络输入
           depth---------------网络深度
           scope---------------网络名称
           reuse---------------网络是否重用
    OUTPUT:net-----------------网络返回值
    ========================================================================'''
    def VDSR(self,inputs,depth = 15,scope = 'VDSR',reuse = None):
        with tf.variable_scope(scope,reuse = reuse):
            with slim.arg_scope([slim.conv2d],activation_fn = None):
                net = inputs
                for i in range(depth-1):
                    net = slim.conv2d(net,64,3,stride = 1,padding = 'SAME',scope = 'conv{}'.format(i))
                    net = tf.nn.leaky_relu(net,alpha = 0.6,name = 'act{}'.format(i))
                net = slim.conv2d(net,1,3,stride = 1,padding = 'SAME',scope = 'conv{}'.format(depth))
                net = net+inputs
                return net

    '''============================= DRCN网络结构 ==============================
    FUNCTION:   DRCN
    FEATURE:    DRCN网络结构
    INPUTS:self,inputs,depth,scope,reuse
           self----------------自身对象
           inputs--------------网络输入
           depth---------------网络深度
           scope---------------网络名称
           reuse---------------网络是否重用
    OUTPUT:net-----------------网络返回值
    ========================================================================'''
    def DRCN(self,inputs,depth = 17,scope = 'DRCN',reuse = None):
        with tf.variable_scope(scope,reuse = reuse):
            with slim.arg_scope([slim.conv2d],activation_fn = None):
                weight_recursive = tf.get_variable(shape = [3,3,256,256],name = 'weight_recursive',dtype = tf.float64)
                H = depth*[None]
                net = slim.conv2d(inputs,256,3,stride = 1,padding = 'SAME',scope = 'conv1')
                net = tf.nn.leaky_relu(net,alpha = 0.6,name = 'act1')
                net = slim.conv2d(net,256,3,stride = 1,padding = 'SAME',scope = 'conv2')
                net = tf.nn.leaky_relu(net,alpha = 0.6,name = 'act2')
                for i in range(2,depth+2):
                    net = tf.nn.conv2d(net,weight_recursive,strides = [1,1,1,1],padding = 'SAME')
                    net = tf.nn.leaky_relu(net,alpha = 0.6,name = 'act{}'.format(i))
                    H[i-2] = net
                W = tf.Variable(np.full(fill_value=1.0 / depth, shape=[depth], dtype=np.float64), name="LayerWeights")
                W_sum = tf.reduce_sum(W)
                weight_reconstruction1 = tf.get_variable(shape = [3,3,256,256],name = 'weight_reconstruction1',dtype = tf.float64)
                weight_reconstruction2 = tf.get_variable(shape = [3,3,256,1],name = 'weight_reconstruction2',dtype = tf.float64)
                output_li = depth*[None]
                for i in range(depth):
                    net = tf.nn.conv2d(H[i],weight_reconstruction1,strides=[1, 1, 1, 1], padding='SAME')
                    net = tf.nn.leaky_relu(net,alpha = 0.6)
                    net = tf.nn.conv2d(net,weight_reconstruction2,strides=[1, 1, 1, 1], padding='SAME')
                    net = tf.nn.leaky_relu(net,alpha = 0.6)
                    H[i] = net#+inputs
                    output_li[i] = H[i]*W[i]/W_sum
                output = tf.add_n(output_li)
                output = output + inputs
                #print(output.shape)
                return output
   
    '''============================= LapSRN网络结构 ============================
    FUNCTION:   LapSRN
    FEATURE:    LapSRN网络结构
    INPUTS:self,inputs,depth,scope,reuse
           self----------------自身对象
           inputs--------------网络输入
           depth---------------网络深度
           scope---------------网络名称
           reuse---------------网络是否重用
    OUTPUT:net-----------------网络返回值
    ========================================================================''' 
    def LapSRN(self,inputs,depth = 10,scope = 'LapSRN',reuse = None):
        def SubpixelConv2d(X,r,n_out_channel):
            bsize, a, b, c = X.get_shape().as_list()
            bsize = tf.shape(X)[0]
            Xs=tf.split(X,r,3)
            Xr=tf.concat(Xs,2)
            X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel))
            return X
        def ElementwiseLayer(layer,combine_fn = tf.minimum):
            outputs = layer[0]
            for l in layer:
                outputs = combine_fn(outputs,l)
            return outputs            
        def LapSRN_block(input_net,net_feature,depth = depth,reuse = None):
            with tf.variable_scope("LapSRN_block",reuse = reuse):
                with slim.arg_scope([slim.conv2d],activation_fn = None):
                    net = net_feature
                    for i in range(depth):
                        net = self.prelu(net,i)
                        net = slim.conv2d(net,64,3,stride = 1,padding = 'SAME')
                    net_feature = ElementwiseLayer(layer = [net_feature,net],combine_fn=tf.add)
                    net_feature = self.prelu(net_feature,depth)
                    net_feature = slim.conv2d(net_feature,256,3,stride = 1,padding = 'SAME')
                    n_out_channel = int(int(net_feature.get_shape()[-1])/(self.zoom**2))
                    net_feature = SubpixelConv2d(net_feature,self.zoom,n_out_channel)
                    grad_level = slim.conv2d(net_feature,1,3,stride = 1,padding = 'SAME')
                    grad_level = tf.nn.leaky_relu(grad_level,alpha = 0.6)
                    net_image = slim.conv2d(input_net,12,3,stride = 1,padding = 'SAME')
                    net_image = tf.nn.leaky_relu(net_image,alpha = 0.6)
                    net_image = SubpixelConv2d(net_image,r = 2,n_out_channel = 1)
                    net_image = ElementwiseLayer(layer = [grad_level,net_image],combine_fn = tf.add)
                    return net_image,net_feature
        with tf.variable_scope(scope,reuse = reuse):
            net_feature = slim.conv2d(inputs,64,3,stride = 1,padding = 'SAME')
            net_image = inputs
            net_image1 = None
            net_image2 = None
            if self.zoom < 4:
                net_image1,net_feature1 = LapSRN_block(net_image,net_feature,reuse = reuse)
                print(net_image1)
                return net_image1
            elif self.zoom == 4:
                net_image1,net_feature1 = LapSRN_block(net_image,net_feature,reuse = reuse)
                net_image2,net_feature2 = LapSRN_block(net_image1,net_feature1,reuse = reuse)
                return net_image2
    
        
    
    """未完成的模型"""
    def DRRN(self):
        return
    
    def MemNet(self):
        return
    
    def IDN(self):
        return
    
    def EDSR_baseline(self):
        return
    
    def SRMDNF(self):
        return
    
    def RCAN(self):
        return
    
    '''========================= get_result返回网络模型 ========================
    FUNCTION:   get_result
    FEATURE:    get_result返回网络模型
    INPUTS:self
           self----------------自身对象
    OUTPUT:网络函数句柄
    ========================================================================'''
    def get_result(self):
        if(self.model == 'SRCNN'):
            return self.SRCNN
        elif(self.model == 'FSRCNN'):
            return self.FSRCNN
        elif(self.model == 'VDSR'):
            return self.VDSR
        elif(self.model == 'DRCN'):
            return self.DRCN
        elif(self.model == 'LapSRN'):
            return self.LapSRN
        elif(self.model == 'DRRN'):
            return self.DRRN
        elif(self.model == 'MemNet'):
            return self.MemNet
        elif(self.model == 'IDN'):
            return self.IDN
        elif(self.model == 'EDSR_baseline'):
            return self.EDSR_baseline
        elif(self.model == 'SRMDNF'):
            return self.SRMDNF
        elif(self.model == 'RCAN'):
            return self.RCAN
        else:
            return None

    '''======================== grenerate_train网络训练 ========================
    FUNCTION:   grenerate_train
    FEATURE:    grenerate_train网络训练
    INPUTS:self,y_train_ph,output
           self----------------自身对象
           y_train_ph----------网络输出占位符
           output--------------网络输出
    OUTPUT:train_op,loss
           train_op------------训练op
           loss----------------损失值
    ========================================================================'''
    def grenerate_train(self,y_train_ph,output):
        loss = tf.losses.mean_squared_error(y_train_ph,output)
        global_step = tf.Variable(0,trainable = False)
        learning_rate = tf.train.exponential_decay(self.lr,
               global_step,self.batch_size,0.9,staircase = True)
        train_op = tf.train.AdamOptimizer(learning_rate)
        grads,v = zip(*train_op.compute_gradients(loss))
        grads,_ = tf.clip_by_global_norm(grads,5)
        train_op = train_op.apply_gradients(zip(grads,v),global_step = global_step)
        return train_op,loss

    '''============================ trainNet训练网络 ===========================
    FUNCTION:   trainNet
    FEATURE:    trainNet训练网络
    INPUTS:self,x_train,y_train,x_test,y_test,train_size,test_size,Epoch,
        iter_view,saved_path,sparse_flag
           self----------------自身对象
           x_train-------------训练数据集输入
           y_train-------------训练数据集输出
           x_test--------------测试数据集输入
           y_test--------------测试数据集输出
           train_size----------训练集大小
           test_size-----------测试集大小
           Epoch---------------训练轮数
           iter_view-----------向屏幕输出的迭代次数
           saved_path----------模型保存路径
           sparse_flag---------是否训练稀疏数据
    OUTPUT:min_index,min_loss
           min_index-----------最小的loss出现的id
           min_loss------------最小的loss值
    ========================================================================'''
    def trainNet(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            train_size,
            test_size,
            Epoch = int(5e3),
            iter_view = 500,
            saved_path = 'Sparsex2'
            ):
        batch_size = self.batch_size
        tf.reset_default_graph()
        x_train_ph = tf.placeholder(shape = (None,16,16,1),dtype = tf.float64)
        y_train_ph = tf.placeholder(shape = (None,16,16,1),dtype = tf.float64)
        x_test_ph = tf.placeholder(shape = (None,16,16,1),dtype = tf.float64)
        y_test_ph = tf.placeholder(shape = (None,16,16,1),dtype = tf.float64)
        func = self.get_result()
        output = func(x_train_ph)
        test_output = func(x_test_ph,reuse = True)
        loss_test = tf.losses.mean_squared_error(y_test_ph,test_output)
        train_op,loss = self.grenerate_train(y_train_ph,output)
        saved_dir = saved_path+'/model.ckpt'
        #开启训练
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        train_loss_li = [];test_loss_li = []
        print('='*74)
        for e in range(Epoch):
            batch1 = np.random.randint(0,train_size,size = batch_size)
            t_x_train = x_train[batch1]
            t_y_train = y_train[batch1]
            batch2 = np.random.randint(0,test_size,size = batch_size)
            t_x_test = x_test[batch2]
            t_y_test = y_test[batch2]
            sess.run(train_op,feed_dict = {x_train_ph:t_x_train,y_train_ph:t_y_train})
            if e % iter_view == iter_view - 1:
                train_loss = sess.run(loss,feed_dict = {x_train_ph:t_x_train,y_train_ph:t_y_train})
                test_loss = sess.run(loss_test,feed_dict = {x_test_ph:t_x_test,y_test_ph:t_y_test})
                print('Epoch:{},train_loss:{:.5f},test_loss:{:.5f}'.format(e+1,train_loss,test_loss))
                saver.save(sess = sess,save_path = saved_dir,global_step = (e+1))
                train_loss_li.append(train_loss);test_loss_li.append(test_loss)
        print('='*30 + 'Train Done !!!' + '='*30)
        print('The min_test is:index--{},loss--{} ; The min_train is:index--{},loss--{}'
                      .format(500*test_loss_li.index(min(test_loss_li))+500,
                              min(test_loss_li),
                              500*train_loss_li.index(min(train_loss_li))+500,
                              min(train_loss_li)))
        print('='*30 + 'Train Done !!!' + '='*30)
        saver.save(sess = sess,save_path = saved_dir)
        sess.close()
        print(x_test_ph)
        print(test_output)
        #train_loss_np = np.array(train_loss_li)
        test_loss_np = np.array(test_loss_li)
        #loss_np = train_loss_np+test_loss_np
        loss_np = test_loss_np
        min_index = np.where(loss_np == min(loss_np))
        min_index = min_index[0][0]*iter_view+iter_view
        min_loss = min(loss_np)
        return min_index,min_loss,train_loss_li,test_loss_li,x_test_ph.name,test_output.name

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
    def prepareSparseData(self,block_im_path,ratio):
            
        saved_x_train = np.load(block_im_path+'xtrain_seq.npy')
        saved_y_train = np.load(block_im_path+'ytrain_seq.npy')
    
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

    '''================== prepareImageData准备图片训练数据 =====================
    FUNCTION:   prepareImageData
    FEATURE:    prepareImageData准备图片训练数据
    INPUTS:self,source_dir,ratio,subimg_h,subimg_w
           self----------------自身对象
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
            self,
            source_dir1,
            source_dir2,
            ratio,
            scale,
            subimg_h = 16,
            subimg_w = 16):
        
        for i in range(int(100*(1-ratio))):
            filename = source_dir1+str(i)+'.png'
            img = Image.open(filename).convert('L')
            img1 = img.resize((512,512))
            #img2 = img1.resize((512//scale,512//scale))
            #img2 = DS.DownScale_n_GaussianBlur_n_AddGaussNoise(img1,(512,512),scale)
            filename2 = source_dir2+str(i)+'.png'
            img2 = Image.open(filename2).convert('L')
            img2 = img2.resize((512,512))
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
        train_img = train_img.reshape((train_img.shape[0],16,16,1))
        label_img = label_img.reshape((label_img.shape[0],16,16,1))
        for i in range(int(100*(1-ratio)),100):
            filename = source_dir1+str(i)+'.png'
            img = Image.open(filename).convert('L')
            img1 = img.resize((512,512))
            #img2 = img1.resize((512//scale,512//scale))
            #img2 = DS.DownScale_n_GaussianBlur_n_AddGaussNoise(img1,(512,512),scale)
            filename2 = source_dir2+str(i)+'.png'
            img2 = Image.open(filename2).convert('L')
            img2 = img2.resize((512,512))
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
        test_img = test_img.reshape((test_img.shape[0],16,16,1))
        test_label_img = test_label_img.reshape((test_label_img.shape[0],16,16,1))
        return train_img,label_img,test_img,test_label_img








