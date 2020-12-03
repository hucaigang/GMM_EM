# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:45:18 2019

@author: 426-2019级-1
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse#做圆
def dataset_select(dataset_name):
    if(dataset_name=='OFD'):
        return pd.read_csv('H://机器学习大作业//GMM(gaussian mixture model)andEM//datasets//faithful.csv')
    elif(dataset_name=='Iris'):
        return pd.read_csv('H://机器学习大作业//GMM(gaussian mixture model)andEM//datasets//iris.csv')
    else:
        return pd.read_csv('H://机器学习大作业//GMM(gaussian mixture model)andEM//datasets//random_points.csv')

def dim_trans(point):#将切片后的numpy数组维度确定，防止出错
    temp_point=np.zeros((point.shape[0],1))
    for i in range(point.shape[0]):
        temp_point[i]=point[i]
    return temp_point
    
def Nor_dis(point,avg,cov,dim):
    temp_point=dim_trans(point)
    power_sign=dim/2
    tmp_mat=temp_point - avg
    tmp_mat1=np.transpose(tmp_mat)
    val=tmp_mat1.dot(np.linalg.inv(cov))
    val=val.dot(tmp_mat)
    result= 1 / (np.power((2*np.pi),power_sign)) / np.sqrt(np.linalg.det(cov)) * np.exp(-val/2)
    return result[0,0]

class GMM():#聚类中心的选择问题。
    def __init__(self,data_set,init_avg_mat,max_iter_times):
        #DATA_PART 
        self.char_num=len(data_set.columns)-1 #记录多少个特征
        self.pts_num=data_set.shape[0] #记录多少数据点
        self.char_name=data_set.columns #特征名
        self.char=[] #保存特征数据值 1x272
        for i in range(self.char_num):
            self.char.append(data_set[self.char_name[i+1]])  
        self.char=np.array(self.char,dtype=np.float32) #list->array
        #ALGORITHM PART(initializing parameters)
        self.iter_times=max_iter_times 
        self.K=len(init_avg_mat)
        self.val_mat=[1/self.K]*self.K # value matrix datatype:list
        self.avg_mat=init_avg_mat ##mean matrix datatype:list[np array]
        self.center_points=[]#初始化
        for i in range(self.K):
            self.center_points.append(self.avg_mat[i])
        self.cov_mat=[np.diag(np.array([2,2]))]*self.K ##covariance matrix datatype:list[np array]
        #postier probability matrix
        self.P=np.zeros((self.K,self.pts_num),dtype=np.float64) #临时存储，放置每个点在每个模型下的值,size(K , n)
        
    def E_step(self):
        for i in range(self.pts_num):#每一个点
            den=0 #归一化
            for j in range(self.K): #对每个模型 update P
                pp=Nor_dis(self.char[:,i],self.avg_mat[j],self.cov_mat[j],self.char_num) 
                den+=(self.val_mat[j]*pp)
                self.P[j,i]=self.val_mat[j]*pp #对于每个点
            for j in range(self.K):
                self.P[j,i]/=den
        
    def M_step(self): #update parameters
        for i in range(self.K):
            self.val_mat[i]=np.mean(self.P[i])
            avg_sum=np.zeros((self.char_num,1),dtype=np.float32)
            cov_sum=np.zeros((self.char_num,self.char_num),dtype=np.float32)
            for j in range(self.pts_num):
                temp_point=dim_trans(self.char[:,j]) #将维度确定
                cov_sum+=self.P[i,j]*(temp_point-self.avg_mat[i]).dot((temp_point-self.avg_mat[i]).transpose())
                avg_sum+=temp_point*self.P[i,j]
            self.avg_mat[i]=avg_sum/np.sum(self.P[i])
            self.cov_mat[i]=cov_sum/np.sum(self.P[i])
            self.center_points.append(avg_sum/np.sum(self.P[i]))
            
    def run(self):
        for i in range(self.iter_times):
            self.E_step()
            self.M_step()
            
    def draw_pic(self):
        fig=plt.figure()
        ax = plt.subplot()
        color_list=['b','g','k','black','c','r'] #中心点颜色选择
        marker_list=['1','.','^','v','2','3'] #中心点图标选择
        if(self.char_num<=2):
            plt.scatter(self.char[0],self.char[1],s=20,c='y') #x eruption y waiting time
            plt.xlabel(self.char_name[1])
            plt.ylabel(self.char_name[2])
            for i in range(self.iter_times+1):#迭代第几次
                for j in range(self.K): #第k个模型的中心点
                    plt.scatter(self.center_points[self.K*i+j][0,0],self.center_points[self.K*i+j][1,0],
                                s=200,c=color_list[j],marker=marker_list[j]) #散点图
            fig.legend([i+1 for i in range(self.K)])
            for i in range(self.K):#开始画置信椭圆
                vals,vecs=np.linalg.eigh(self.cov_mat[i])
                x,y=vecs[:,1]
                theta = np.degrees(np.arctan2(y,x))
                h,w=np.sqrt(5.991*vals)  #0.95置信度(可以修改)
                ell=Ellipse(self.avg_mat[i],2*w,2*h,theta,color=color_list[5-i],linewidth=2,linestyle='--')
                ell.set_alpha(0.5)
                ell.set_fill(False)
                ax.add_artist(ell)
            plt.show()
        
if __name__=="__main__":
    dataset_name=str(input('OFD or Iris or nothing'))
    dataset=dataset_select(dataset_name)
    if dataset_name=='OFD':
        #对于Old Faithful dataset(right)
        init_mu_mat=[np.array([[1],[30]]),np.array([[4],[90]])]
        gmm1=GMM(dataset,init_mu_mat,30)
        gmm1.run()
        gmm1.draw_pic()
    elif dataset_name=='Iris':
    #对于Iris
        index=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']#因为只能做两个特征的可视化，所以需要选择去除哪些特征
        dataset.drop([index[0],index[1],index[4]],axis=1,inplace=True)#删除第2、4、5个特征
        init_mu_mat=[np.array([[4],[2]]),np.array([[3],[1.1]]),np.array([[2],[1.2]])] #初始均值矩阵 (手动给定)对于Iris
        gmm1=GMM(dataset,init_mu_mat,30) #修改参数 (数据集，初始均值矩阵，迭代次数)
        gmm1.run()
        gmm1.draw_pic()
    else:
        #对于自建数据集
        init_mu_mat=[np.array([[3],[3]]),np.array([[20],[20]]),np.array([[90],[90]])]
        gmm1=GMM(dataset,init_mu_mat,50)
        gmm1.run()
        gmm1.draw_pic()
    
