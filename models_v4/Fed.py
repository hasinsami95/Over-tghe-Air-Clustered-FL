#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn

def FedAvg_noscale(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedMult(w,sc):
    w_scaled = copy.deepcopy(w)
    for k in w_scaled.keys():
#         for i in range(1,len(w)):
#             w_scaled[k] = sc*w_scaled[k]
        w_scaled[k] = torch.mul(w_scaled[k], sc)
    return w_scaled

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
def FedAvg3(w,scale):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k]*scale[0]
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]*scale[i]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg2(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] =torch.div(w_avg[k],45)
    for k in w_avg.keys():
        for i in range(1,len(w)):
            if (i<5):
                w_avg[k] += torch.div(w[i][k],45)
            else:
                w_avg[k] += torch.div(w[i][k],(45/2))
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAdd(w1,w2):
    w_final=copy.deepcopy(w1)
    #print(w2)
    for k in w_final.keys():
        w_final[k]=w1[k].to(torch.device("cuda:3"))+w2[k].to(torch.device("cuda:3"))
    return w_final


def FedSubstract(w1,w2):
    w_final=copy.deepcopy(w1)
    #print(w2)
    for k in w_final.keys():
        w_final[k]=w1[k].to(torch.device("cuda:3"))-w2[k].to(torch.device("cuda:3"))
    return w_final



def FedAvg_vectorization(w):
    vect=[]
    mat=[]
    dimension=[]
    w_avg = copy.deepcopy(w[0])
    count=0
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
        mat.append((w_avg[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    
    return vect,dimension

def FedAvg_gradient(g):
    g_avg=copy.deepcopy(g[0])
    for i in range(1,len(g)):
        g_avg+=g[i]
    g_avg=g_avg/len(g)
    return g_avg
    
def FedAvg_gradient2(g):
    g_avg=copy.deepcopy(g[0])
    for i in range(1,len(g)):
        g_avg+=g[i]
    #g_avg=g_avg/len(g)
    return g_avg

def weight_vectorization(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    for k in w.keys():
    
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
    #or return vect[0] depending on dimension (1 D or 2 D)
    
def weight_vectorization_gen(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    feature_vect=np.array([])
    for k in w.keys():
    #for k in range(len(w)):
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        feature_vect=np.concatenate((feature_vect,vect[count]),axis=None)
        count=count+1
    #feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
    
def weight_vectorization_gen2(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    feature_vect=np.array([])
    #for k in w.keys():
    for k in range(len(w)):
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        feature_vect=np.concatenate((feature_vect,vect[count]),axis=None)
        count=count+1
    #feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension

def weight_vectorization_femnist(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    for k in w.keys():
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3],vect[4],vect[5]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
def weight_vectorization_cifar(w):
    #w=w.to(torch.device("cpu"))
    vect=[]
    mat=[]
    dimension=[]
    count=0
    for k in w.keys():
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3],vect[4],vect[5],vect[6],vect[7],vect[8],vect[9],vect[10],vect[11],vect[12],
                                 vect[13],vect[14],vect[15]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
    
def vectorization(w):
    vect=[]
    mat=[]
    dimension=[]
    w_avg = copy.deepcopy(w)
    count=0
    for k in w_avg.keys():
        mat.append((w_avg[k]).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    return vect,dimension
        

