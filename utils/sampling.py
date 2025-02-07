#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random



def mnist_iid_cluster(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    #num_items=2000 for paper 2
    num_items=1200 # no. of users =5 for each cluster, per user 1200 data
    #num_items_test=300
    dict_users,dict_users_test, all_idxs = [],[],[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    #for i in range(num_users):
        #dict_users_test.append(np.random.choice(dict_users[i],num_items_test,replace=False))
        #dict_users[i]=list(set(dict_users[i])-set(dict_users_test[i]))
    
    return dict_users

def mnist_iid_cluster_10users(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    #num_items=2000 for paper 2
    num_items=600 # no. of users =10 for each cluster, per user 600 data
    #num_items_test=300
    dict_users,dict_users_test, all_idxs = [],[],[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    #for i in range(num_users):
        #dict_users_test.append(np.random.choice(dict_users[i],num_items_test,replace=False))
        #dict_users[i]=list(set(dict_users[i])-set(dict_users_test[i]))
    
    return dict_users





def mnist_noniid_cluster(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    #num_items=1100 # no. of users =5 for each cluster, per user 1200 data
    #num_items_test=300
    num_items=2000
    dict_users,dict_users_test, all_idxs = [],[], [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    #for i in range(num_users):
        #dict_users_test.append(np.random.choice(dict_users[i],num_items_test,replace=False))
        #dict_users[i]=list(set(dict_users[i])-set(dict_users_test[i]))
    #return dict_users,dict_users_test
    return dict_users



def cifar_iid_cluster(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    num_items=1000 # no. of users =5 for each cluster, per user 1200 data
    #num_items_test=300
    np.random.seed(10)
    dict_users,dict_users_test, all_idxs = [],[], [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    #for i in range(num_users):
        #dict_users_test.append(np.random.choice(dict_users[i],num_items_test,replace=False))
        #dict_users[i]=list(set(dict_users[i])-set(dict_users_test[i]))
    
    return dict_users

def cifar_iid_cluster_MI(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    num_items=500 # no. of users =5 for each cluster, per user 1200 data
    #num_items_test=300
    np.random.seed(10)
    dict_users,dict_users_test, all_idxs = [],[], [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    #for i in range(num_users):
        #dict_users_test.append(np.random.choice(dict_users[i],num_items_test,replace=False))
        #dict_users[i]=list(set(dict_users[i])-set(dict_users_test[i]))
    return dict_users

def cifar_noniid_cluster(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    num_items=2000 # no. of users =5 for each cluster, per user 1200 data
    np.random.seed(10)
    dict_users,all_idxs = [], [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    
    return dict_users
    
    
def cifar_noniid_cluster_varying_users(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(10)
    num_items = int(len(dataset)/num_users)
    #num_items=2000 # no. of users =5 for each cluster, per user 1200 data
    dict_users,all_idxs = [], [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    
    return dict_users
    
def cifar_dirichlet_varying_users_class(dataset, num_users, cluster_no):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(10)
    num_items = int(len(dataset)/num_users)
    train_data_class1=[]
    train_data_class0=[]
    for j in range(cluster_no):
        train_data_class0.append([])
        train_data_class1.append([])
    all_idx0=[]
    all_idx1=[]
    for j in range(len(dataset)):
        data,label=dataset[j]
        if (label==0):
            train_data_class0[0].append(dataset[j])
            all_idx0.append(j)
        elif (label==2):
            train_data_class0[1].append(dataset[j])
            all_idx0.append(j)
        elif (label==4):
            train_data_class0[2].append(dataset[j])
            all_idx0.append(j)
        elif (label==6):
            train_data_class0[3].append(dataset[j])
            all_idx0.append(j)
        elif (label==8):
            train_data_class0[4].append(dataset[j])
            all_idx0.append(j)
    for j in range(len(dataset)):
        data,label=dataset[j]
        if (label==1):
            train_data_class1[0].append(dataset[j])
            all_idx1.append(j)
        elif (label==3):
            train_data_class1[1].append(dataset[j])
            all_idx1.append(j)
        elif (label==5):
            all_idx1.append(j)
            train_data_class1[2].append(dataset[j])
        elif (label==7):
            train_data_class1[3].append(dataset[j])
            all_idx1.append(j)
        elif (label==9):
            train_data_class1[4].append(dataset[j])
            all_idx1.append(j)
            
    
    alpha=[]
    for i in range(num_users):
       alpha.append(20)
    np.random.seed(30)
    p_0=np.random.dirichlet(alpha)
    np.random.seed(25)
    p_1=np.random.dirichlet(alpha)
    len0=len(all_idx0)
    len1=len(all_idx1)
    #num_items=2000 # no. of users =5 for each cluster, per user 1200 data
    dict_users = []
    for i in range(num_users):
        dict_users.append([])
    np.random.seed(10)
    for i in range(num_users):
        num_items=int(p_0[i]*len0)
        a=np.random.choice(all_idx0, num_items, replace=False)
        #dict_users[i].append(np.random.choice(all_idx0, num_items, replace=False))
        num_items=int(p_1[i]*len1)
        b=np.random.choice(all_idx1, num_items, replace=False)
        #dict_users[i].append(np.random.choice(all_idx1, num_items, replace=False))
        dict_users[i]=np.concatenate((a,b),axis=0)
        all_idx0 = list(set(all_idx0) - set(dict_users[i]))
        all_idx1 = list(set(all_idx1) - set(dict_users[i]))
        random.shuffle(dict_users[i])
    
    return dict_users
    
def cifar_dirichlet_varying_users_class_v2(dataset, num_users, cluster_no):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(10)
    num_items = int(len(dataset)/num_users)
    train_data_class1=[]
    train_data_class0=[]
    for j in range(cluster_no):
        train_data_class0.append([])
        train_data_class1.append([])
    all_idx0=[]
    all_idx1=[]
    for j in range(len(dataset)):
        data,label=dataset[j]
        if (label==0):
            train_data_class0[0].append(dataset[j])
            all_idx0.append(j)
        elif (label==2):
            train_data_class0[1].append(dataset[j])
            all_idx0.append(j)
        elif (label==4):
            train_data_class0[2].append(dataset[j])
            all_idx0.append(j)
        elif (label==6):
            train_data_class0[3].append(dataset[j])
            all_idx0.append(j)
        elif (label==8):
            train_data_class0[4].append(dataset[j])
            all_idx0.append(j)
    for j in range(len(dataset)):
        data,label=dataset[j]
        if (label==1):
            train_data_class1[0].append(dataset[j])
            all_idx1.append(j)
        elif (label==3):
            train_data_class1[1].append(dataset[j])
            all_idx1.append(j)
        elif (label==5):
            all_idx1.append(j)
            train_data_class1[2].append(dataset[j])
        elif (label==7):
            train_data_class1[3].append(dataset[j])
            all_idx1.append(j)
        elif (label==9):
            train_data_class1[4].append(dataset[j])
            all_idx1.append(j)
            
    
    alpha=[]
    for i in range(num_users):
       alpha.append(0.5)
    np.random.seed(10)
    p_0=np.random.dirichlet(alpha)
    np.random.seed(25)
    p_1=np.random.dirichlet(alpha)
    len0=len(all_idx0)
    len1=len(all_idx1)
    #num_items=2000 # no. of users =5 for each cluster, per user 1200 data
    dict_users = []
    for i in range(num_users):
        dict_users.append([])
    np.random.seed(10)
    for i in range(num_users):
        num_items=int(p_0[i]*len0)
        a=np.random.choice(all_idx0, num_items, replace=False)
        #dict_users[i].append(np.random.choice(all_idx0, num_items, replace=False))
        num_items=int(p_1[i]*len1)
        b=np.random.choice(all_idx1, num_items, replace=False)
        #dict_users[i].append(np.random.choice(all_idx1, num_items, replace=False))
        dict_users[i]=np.concatenate((a,b),axis=0)
        all_idx0 = list(set(all_idx0) - set(dict_users[i]))
        all_idx1 = list(set(all_idx1) - set(dict_users[i]))
        random.seed(10)
        random.shuffle(dict_users[i])
    
    return dict_users
    
        
        
def femnist_noniid_cluster(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    num_items=800 # no. of users =5 for each cluster, per user 800 data
    dict_users,all_idxs = [], [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users.append(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    
    return dict_users


#if __name__ == '__main__':
    #dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),
                                       #transforms.Normalize((0.1307,), (0.3081,))]))
    #num = 100
    #d = mnist_noniid(dataset_train, num)
