# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:23:01 2020

@author: 月光下的云海
"""

import numpy as np
import pandas as pd

#字典类       
class Dictionary():
    
    #**********************************************
    #FUNCTION:__init__
    #FEATURE: 将Dictionary里面的变量初始化
    #参数说明：
    #   self-------自身对象
    #   dic--------字典
    #   id---------列标
    #   centroids--聚类中心
    #返回值：
    #   无
    #**********************************************
    def __init__(self,dic,id,centroids):
        self.dic = dic#字典
        self.id = id#标号
        self.centroids = centroids#聚类中心

        
    #**********************************************
    #FUNCTION:__init__
    #FEATURE: 将Dictionary里面的变量初始化
    #参数说明：
    #   self-------自身对象
    #返回值：
    #   df---------一个dataframe包含Dictionary的所有信息
    #**********************************************
    def convert_to_dataframe(self):
        size = self.centroids.shape[0]
        id = pd.DataFrame({
                  'id'+str(self.id):{0:self.id}
                  })

        t_s = self.centroids
        t_s = t_s.reshape(size,)
        t_s = np.array(t_s)
        t_s = t_s.reshape(size,)
        centroids = pd.DataFrame({
                                  'centroids'+str(self.id):t_s
                                  })
        dic = pd.DataFrame(self.dic)
        df = pd.concat([id,centroids,dic],axis=1)
        return(df)