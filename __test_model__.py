# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:28:58 2020

@author: 月光下的云海
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from DLL.Valuation import psnr
from DLL.DictionaryLearning import pca_ISTA
from skimage.measure import compare_ssim as ssim

'''稀疏模型的超分测试'''
def sparse_super_resolution(zoom_times,best_index,filename,model_path,version,x_train_ph_name,output_name):

    new_image = Image.open(filename)
    new_image = new_image.resize((512,512))
    new_image = new_image.resize((512//zoom_times,512//zoom_times))
    new_image = new_image.resize((512,512))
    new_image = new_image.convert("YCbCr")
    new_image = np.asarray(new_image)
    
    y_new_image = new_image[:,:,0]
    cb_new_image = new_image[:,:,1]
    cr_new_image = new_image[:,:,2]
    #N_image,bloc_im_li,dic_li = pca_ISTA(y_new_image,16,1,flag = False)
    _,bloc_im_li,dic_li = pca_ISTA(y_new_image,16,1,flag = False)
    xtrain_seq = np.zeros((0,16,16));label_li = [];
    for row in range(32):
        for col in range(32):
            hr_sparse = bloc_im_li[row][col].sparse_code
            hr_sparse = hr_sparse.reshape((16,16),order = 'F')
            hr_sparse = hr_sparse.reshape((1,16,16))
            xtrain_seq = np.vstack((xtrain_seq,hr_sparse))
            label_li.append(bloc_im_li[row][col].set)
 
    xtrain_seq = xtrain_seq.reshape((1024,16,16,1))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('F:/'+version+'/'+model_path+'/model.ckpt-{}.meta'.format(best_index))
    saver.restore(sess, tf.train.latest_checkpoint('F:/'+version+'/'+model_path+'/'))
    graph = tf.get_default_graph()
    #output = graph.get_tensor_by_name('SRCNN_1/conv3/BiasAdd:0')
    output = graph.get_tensor_by_name(output_name)
    #x_train_ph = graph.get_tensor_by_name('Placeholder_2:0')
    x_train_ph = graph.get_tensor_by_name(x_train_ph_name)
    outputs = sess.run(output,feed_dict = {x_train_ph:xtrain_seq})
    sess.close()
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
    
    new_image = np.stack((new_image,cb_new_image,cr_new_image),axis = 2)
    new_image = np.asarray(Image.fromarray(np.uint8(new_image),"YCbCr").convert("RGB"))
    
    or_image = Image.open(filename).convert("RGB")
    or_image = or_image.resize((512,512))
    or_image = np.asarray(or_image)
    Ipsnr = psnr(or_image,new_image)
    Issim = ssim(or_image,new_image,multichannel=True)
    print('The quality reconstructed image,(PSNR:{:.5f},SSIM:{:.5f})'.format(Ipsnr,Issim))
    return Ipsnr,Issim,new_image
    
'''网络模型的超分测试'''
def net_super_resolution(zoom_times,filename,model_path,version,x_train_ph_name,output_name):
    or_image = Image.open(filename).convert("RGB")
    or_image = or_image.resize((512,512))
    lr_image = or_image.resize((512//zoom_times,512//zoom_times))
    lr_image = lr_image.resize((512,512))
    lr_image = lr_image.convert("YCbCr")
    or_image = np.asarray(or_image)
    
    lr_image = np.asarray(lr_image)
    
    cb_lr_image = lr_image[:,:,1]
    cr_lr_image = lr_image[:,:,2]
    lr_image = lr_image[:,:,0]
    
    lr_image_seq = np.zeros((0,16,16))
    for i in range(32):
        for j in range(32):
            block = lr_image[16*i:16*(i+1),16*j:16*(j+1)]
            block = block.reshape((1,16,16))
            lr_image_seq = np.vstack((lr_image_seq,block))
    lr_image_seq = lr_image_seq.reshape((1024,16,16,1))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('F:/'+version+'/'+model_path+'/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('F:/'+version+'/'+model_path+'/'))
    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name(output_name)
    x_train_ph = graph.get_tensor_by_name(x_train_ph_name)
    outputs = sess.run(output,feed_dict = {x_train_ph:lr_image_seq})
    sess.close()
    new_image = 255*np.ones((512,512))
    for i in range(32):
        for j in range(32):
            block = np.squeeze(outputs[32*i+j])
            new_image[16*i:16*(i+1),16*j:16*(j+1)] = block
    
    new_image = np.stack((new_image,cb_lr_image,cr_lr_image),axis = 2)
    new_image = np.array(Image.fromarray(np.uint8(new_image),"YCbCr").convert("RGB"))
    Ipsnr = psnr(or_image,new_image)
    Issim = ssim(or_image,new_image,multichannel=True)
    print('The quality reconstructed image,(PSNR:{:.5f},SSIM:{:.5f})'.format(Ipsnr,Issim))
    return Ipsnr,Issim,new_image

if __name__ == '__main__':
    
    Based_Model = 'FSRCNN'
    amplification = 1
    summary = open(Based_Model+'_Summary.txt','r')
    for line in summary:
        try:
            line.index(Based_Model)
        except ValueError:
            print('记录不存在!!')
        else:
            line = line.replace('\n','')
            loc = line.index(':')
            rloc = line.index(',')
            best_index = int(line[loc+1:rloc])
            loc = line.index('Input:')
            rloc = line.index(',Output:')
            x_train_ph_name = line[loc:rloc]
            x_train_ph_name = x_train_ph_name.replace('Input:','')
            output_name = line[rloc:]
            output_name = output_name.replace(',Output:','')
    summary.close()
    version_file = os.path.abspath(os.path.dirname(os.getcwd()))
    version = version_file.replace('F:\\','')
    test_sr_data_file = 'F:/'+version+'/BM3D/0.1/'
    test_result_Base_dir = 'F:/'+version+'/'+Based_Model+'_Result/'
    test_result_Sp_dir = 'F:/'+version+'/'+Based_Model+'_SparseSR_Result/'
    Report = open(Based_Model+'_Report_x{}.txt'.format(amplification),'w+')
    subdir = [subdir for _, subdir, _ in os.walk(test_sr_data_file)][0]
    Total_SRCNNpsnr_li = []
    Total_SRCNNssim_li = []
    Total_NET_PCApsnr_li = []
    Total_NET_PCAssim_li = []
    Total_NET_time_li = []
    from time import time
    for file0 in subdir:
        print('\n'+'='*81 +'\n'+'|'+' '*20+'Process of super resolution on '+file0+' '*20+'|'+'\n'+'='*81 )
        image_path = test_sr_data_file+file0+'/'
        filename_li = [name for _,_,name in os.walk(image_path)][0]
        SRCNNpsnr_li = []
        SRCNNssim_li = []
        NET_PCApsnr_li = []
        NET_PCAssim_li = []
        NET_time_li = []
        if not os.path.exists(test_result_Sp_dir+file0):
            os.makedirs(test_result_Sp_dir+file0)
        if not os.path.exists(test_result_Base_dir+file0):
            os.makedirs(test_result_Base_dir+file0)
        for file in filename_li:
            path = image_path+file
            print('='*25+' the '+str(file)+' image '+'='*25)
            t = time()
            
            Proposedpsnr,Proposedssim,sr_im = sparse_super_resolution(zoom_times = amplification,
                                                                best_index = best_index,
                                                                filename = path,
                                                                model_path = Based_Model+'_SparseSR_x'+str(amplification),
                                                                version = version,
                                                                x_train_ph_name = x_train_ph_name,
                                                                output_name = output_name)
            Image.fromarray(sr_im).save(test_result_Sp_dir+file0+'/'+file)
            t = time()-t
            print('Time elapsed:',t)
            
            #Proposedpsnr,Proposedssim = 0,0
            
            SRCNNpsnr,SRCNNssim,sr_im = net_super_resolution(zoom_times = amplification,
                                                         filename = path,
                                                         model_path = Based_Model+'x'+str(amplification),
                                                         version = version,
                                                         x_train_ph_name = x_train_ph_name,
                                                         output_name = output_name)
            Image.fromarray(sr_im).save(test_result_Base_dir+file0+'/'+file)
            
            #SRCNNpsnr,SRCNNssim = 0,0
            
            SRCNNpsnr_li.append(SRCNNpsnr)
            SRCNNssim_li.append(SRCNNssim)
            NET_PCApsnr_li.append(Proposedpsnr)
            NET_PCAssim_li.append(Proposedssim)
            NET_time_li += [t]
        report_line = file0+','+Based_Model+'_psnr:{} ,'.format(np.mean(SRCNNpsnr_li))+\
                        Based_Model+'_ssim:{} , '.format(np.mean(SRCNNssim_li)) + \
                        Based_Model + '_PCA_psnr:{} ,'.format(np.mean(NET_PCApsnr_li)) + \
                        Based_Model + '_PCA_ssim:{} \n'.format(np.mean(NET_PCAssim_li))
        Total_SRCNNpsnr_li.append(SRCNNpsnr_li)
        Total_SRCNNssim_li.append(SRCNNssim_li)
        Total_NET_PCApsnr_li.append(NET_PCApsnr_li)
        Total_NET_PCAssim_li.append(NET_PCAssim_li)
        Total_NET_time_li.append(NET_time_li)
        print('\n\n'+report_line+'\n\n')
        Report.write(report_line)
        print('='*82)
    Report.close()











