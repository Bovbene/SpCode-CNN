# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:36:56 2021

@author: 月光下的云海
"""

import tensorflow as tf

class BaseModel():
    
    def __init__(self):
        pass
    
    def lrelu(self,x,alpha = 0.6,name = 'lrelu'):
        return tf.nn.leaky_relu(x,alpha = alpha,name = name)
    
    def prelu(self,x):
        alphas = tf.get_variable(x.get_shape()[-1],initializer=tf.constant_initializer(0.0),dtype=tf.float64)
        pos = tf.nn.relu(x)
        neg = alphas * (x-abs(x))*0.5
        return pos+neg

    def grenerate_train(self,y_train_ph,output):
        loss = self.l1_loss(y_train_ph,output)
        global_step = tf.Variable(0,trainable = False)
        learning_rate = tf.train.exponential_decay(self.lr,
               global_step,self.batch_size,0.9,staircase = True)
        train_op = tf.train.AdamOptimizer(learning_rate)
        grads,v = zip(*train_op.compute_gradients(loss))
        grads,_ = tf.clip_by_global_norm(grads,5)
        train_op = train_op.apply_gradients(zip(grads,v),global_step = global_step)
        return train_op,loss
    
    def l1_loss(self,x,y):
        return tf.reduce_mean(tf.abs(x-y))
    