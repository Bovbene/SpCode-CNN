# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:10:16 2020

@author: 月光下的云海
"""

from DLL.CreateModel import CreateModel
import os
'''===================== WriteToSummary把训练信息写入txt =======================
FUNCTION:   WriteToSummary
FEATURE:    WriteToSummary把训练信息写入txt
INPUTS:model,min_index,min_loss
       model-----------模型名称
       min_index-------最小loss的id
       min_loss--------最小的loss值
OUTPUT:无
============================================================================='''
def WriteToSummary(model,min_index,min_loss,x_test_ph_name,test_output_name):
    line = model+':'+str(min_index)+','+str(min_loss)+'\n'
    if( not os.path.exists(model+'_Summary.txt')):
        Summary = open(model+'_Summary.txt','w')
        Summary.close()
    Summary = open(model+'_Summary.txt','r+')
    summary_content = ''
    try:                                  
        for info in Summary:
            name_loc = info.index(model)
            name_loc = info.index(':')
            name = info[0:name_loc]
            if(model == name):
                info = info.replace(info[name_loc+1:],str(min_index)+','+str(min_loss))
                info = info+';Input:'+x_test_ph_name+',Output:'+test_output_name+'\n'
                summary_content += info
            else:
                summary_content += info
                Summary.close()
        Summary = open(model+'_Summary.txt','w+')
        Summary.write(summary_content)
        Summary.close()
    except ValueError:
        Summary.close()
        Summary = open(model+'_Summary.txt','a+')
        Summary.write(line)
        Summary.close()


if __name__ == '__main__':

    model = input("Which model do u wanna choose :")
    scale = int(input("And the magnification is :"))
    source_dir = os.path.abspath(os.path.dirname(os.getcwd()))+'\\'
    net_model = CreateModel(model = model,lr = 1e-3,batch_size = 128)
    x_test,y_test,x_train,y_train,train_size,test_size = net_model.prepareSparseData(
            source_dir+'Saprse_Train_Data\\',0.2)
    print('\n\nSparseModel ( ' + model +' x '+str(scale)+ ' ) Trainning ... ...')
    
    min_index,min_loss,sp_train_li,sp_test_li,x_test_ph_name1,test_output_name1 = net_model.trainNet(x_train,
                                                                                                   y_train,
                                                                                                   x_test,
                                                                                                   y_test,
                                                                                                   train_size,
                                                                                                   test_size,
                                                                                                   Epoch = int(10e3),
                                                                                                   iter_view = 500,
                                                                                                   saved_path = source_dir+model+'_SparseSR_x'+str(scale))
    WriteToSummary(model,min_index,min_loss,x_test_ph_name1,test_output_name1)
    
    
    train_img,label_img,test_img,test_label_img = net_model.prepareImageData(
            source_dir1 = source_dir+'\\xTrainData\\',source_dir2 = source_dir+'\\yTrainData\\',ratio = 0.2,scale = scale)
    net_model = CreateModel(model = model,lr = 1e-3,batch_size = 128)
    print('\n\nSRModel (' + model +' x '+str(scale)+ ') Trainning ... ...')
    min_index,min_loss,sr_train_li,sr_test_li,x_test_ph_name2,test_output_name2 = net_model.trainNet(train_img,
                                                                                                   label_img,
                                                                                                   test_img,
                                                                                                   test_label_img,
                                                                                                   train_size = train_img.shape[0],
                                                                                                   test_size = test_img.shape[0],
                                                                                                   Epoch = int(5e3),
                                                                                                   iter_view = 500,
                                                                                                   saved_path = source_dir+model+'x'+str(scale))
    
    import matplotlib.pyplot as plt
    plt.plot(sp_test_li,'r');plt.plot(sr_test_li,'b');
    plt.xlabel('Epoch:100')
    plt.ylabel('Loss:0.01')
    plt.title('Loss Curve')
    