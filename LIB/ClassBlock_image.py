# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:14:56 2020

@author: 月光下的云海
"""

import numpy as np
import pandas as pd

class BlockImage():
    def __init__(self,sub_image = None,row_stretch = None,idx = None,sparse_code = None):
        self.sub_image = sub_image
        self.row_stretch = row_stretch
        self.idx = idx
        self.sparse_code = sparse_code

#图像块类
class Block_image():
    
    path = 'F:\\Project2\\Saprse_Train_Data\\'
    
    #**********************************************
    #FUNCTION:__init__
    #FEATURE: 将Block_image里面的变量初始化
    #参数说明：
    #   self-------自身对象
    #   row--------行标
    #   column-----列标
    #   sub_image--子图像块
    #   size_block--图像块大小
    #返回值：
    #   无
    #**********************************************
    def __init__(self,row,column,sub_image,size_block):
        self.row = row#行标
        self.column = column#列标
        self.sub_image = sub_image#子图像块
        self.set = 0#类别标号
        self.sparse_code = np.zeros((size_block*size_block,1))
        self.sparse_code = np.matrix(self.sparse_code)#本图像块的稀疏编码
        self.row_stretch = np.transpose(sub_image)#图像块列拉直
        self.row_stretch = self.row_stretch.reshape((size_block*size_block,1))#图像块列拉直

    
    def save_block(self):
        name1 = str(self.row)
        name2 = str(self.column)
        name3 = str(self.set)
        np.save(self.path+name1+'_'+name2+'_'+name3,self.sparse_code)
    
    #**********************************************
    #FUNCTION:compress_sparese_code
    #FEATURE: 将Bsparse_code压缩，只保存位置和数值
    #参数说明：
    #   self-------自身对象
    #返回值：
    #   无
    #**********************************************
    def compress_sparese_code(self):
        n_sparse_code = np.array([])
        n_pos_code = np.array([])
        ind = 0
        for i in self.sparse_code:
            if(i != 0):
                i = float(i)
                n_sparse_code = np.append(n_sparse_code,i)
                n_pos_code = np.append(n_pos_code,ind)
            ind = ind+1
        self.sparse_code = np.vstack((n_pos_code,n_sparse_code))
        self.sparse_code = np.transpose(self.sparse_code)

    #**********************************************
    #FUNCTION:convert_to_dataframe
    #FEATURE: 将Block_image转化为dataframe
    #参数说明：
    #   self-------自身对象
    #返回值：
    #   df---------一个dataframe包含Block_image的所有信息
    #**********************************************
    def convert_to_dataframe(self):
        size = self.sparse_code.shape[0]
        row_col_set = pd.DataFrame({
                  'row'+str(self.row)+'_'+str(self.column):{ 0: self.row},
                  'column'+str(self.row)+'_'+str(self.column):{0:self.column},
                  'set'+str(self.row)+'_'+str(self.column):{0:self.set}
                  })
        if(self.sparse_code.shape[1] == 1):
            t_s = self.sparse_code
            t_s = t_s.reshape(size,)
            t_s = np.array(t_s)
            t_s = t_s.reshape(size,)        
            code_stretch = pd.DataFrame({
                    'sparse_code'+str(self.row)+'_'+str(self.column):t_s
                    })
        elif(self.sparse_code.shape[1] == 2):
            t_s1 = self.sparse_code[:,0]
            t_s1 = t_s1.reshape(size,)
            t_s1 = np.array(t_s1)
            t_s1 = t_s1.reshape(size,) 
            t_s2 = self.sparse_code[:,1]
            t_s2 = t_s2.reshape(size,)
            t_s2 = np.array(t_s2)
            t_s2 = t_s2.reshape(size,) 
            code_stretch = pd.DataFrame({
                    'sparse_code_pos'+str(self.row)+'_'+str(self.column):t_s1,
                    'sparse_code_val'+str(self.row)+'_'+str(self.column):t_s2
                    })
        df = pd.concat([row_col_set,code_stretch],axis=1)
        return(df)