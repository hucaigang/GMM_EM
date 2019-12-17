# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:53:48 2019

@author: Administrator
"""

import numpy as np

mu1=[4,4]
mu2=[30,30]
mu3=[70,70]

sigma1=[[20,4],[4,1]]
sigma2=[[20,10],[10,12]]
sigma3=[[32,18],[18,19]]

points=[]
points.append(np.random.multivariate_normal(mu1,sigma1,1000)) #点数
points.append(np.random.multivariate_normal(mu2,sigma2,1000))
points.append(np.random.multivariate_normal(mu3,sigma3,1000))

file=open(r'C:\Users\426-2019级-1\Desktop\random_points.csv','w+')
file.write(' ,x,y\n')
for i in range(3):
    for j in range(points[i].shape[0]):
        file.write('%d,%f,%f\n'%(i*1000+j,points[i][j,0],points[i][j,1]))
