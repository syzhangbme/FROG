import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import numpy as np
import pandas as pd
from random import sample
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os
from scipy import stats
import platform
plat = platform.system().lower()

# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)


def getCommonAdj(id_train, adj_type, adj_thr, adj_norm):
    r"""
    计算共同的邻接矩阵，计算方法是所有皮尔逊相关性绝对值矩阵按位相加再卡阈值
    """
    assert adj_norm==False

    all_adj_coef = []
    for index in range(len(id_train)):
        subDirPath = id_train[index]

        for subFileName in os.listdir(subDirPath): # 遍历被试文件夹

            if plat == 'windows':
                subFileName = subFileName.encode('utf-8').decode('utf-8')
            elif plat == 'linux':
                subFileName = subFileName.decode('utf-8')
            
            if adj_type=='pearson' and 'pearson_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)
                all_adj_coef.append(adj_coef)
            
            if adj_type=='spearman' and 'spearman_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)
                all_adj_coef.append(adj_coef)
            
            if adj_type=='cos' and 'cos_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)
                all_adj_coef.append(adj_coef)
            
            if adj_type=='dcor' and 'dcor_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)
                all_adj_coef.append(adj_coef)

    # DCRNN 和 STGCN在这里都不做 加单位矩阵、 归一化

    all_mean_adj_coef = np.mean(np.array(all_adj_coef), axis=0)

    if adj_thr != 'pos' and adj_norm == False: # GAT
        all_mean_vec_coef = all_mean_adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
        thr_value = np.percentile(all_mean_vec_coef, 100-adj_thr)
        all_mean_adj_coef[all_mean_adj_coef<thr_value] = 0.0

    elif adj_thr == 'pos' and adj_norm == False: # GAT
        all_mean_adj_coef[all_mean_adj_coef<0.0] = 0.0

    return all_mean_adj_coef


def getAllData(dataPath='../data/processed/', sitenameList=['site2_ADNI3'], categoryDict={'CN':0, 'MCI':1, 'AD':2}):
    r"""
    单或多中心数据混合, 一起载入
    """

    all_site_id = []
    all_site_label = []

    for sitename in sitenameList:
        site_id = []
        site_label = []

        sitepath = dataPath + sitename

        for subDir in os.listdir(sitepath):
            subDirPath = sitepath + '/' + subDir

            temp_label_path = subDirPath + '/' + 'label.txt'
            temp_label = 'NULL'
            with open(temp_label_path) as f: # 此处读入标签是为了下面等比例划分训练集和测试集
                temp_label = f.read()
            if temp_label in categoryDict.keys():

                site_id.append(subDirPath)
                site_label.append(categoryDict[temp_label])

        all_site_id += site_id
        all_site_label += site_label
    
    all_site_id = np.array(all_site_id)
    all_site_label = np.array(all_site_label)

    return (all_site_id, all_site_label)



def splitTrainTest(test_size=0.2, dataPath='../data/processed/', sitenameList=['site2_ADNI3'], categoryDict={'CN':0, 'MCI':1, 'AD':2}, random_state=0):
    r"""
    单或多中心数据混合, 划分训练集和测试集
    """

    all_site_id, all_site_label = getAllData(dataPath=dataPath, sitenameList=sitenameList, categoryDict=categoryDict)

    # 划分训练集和测试集
    id_train, id_test, label_train, label_test = train_test_split(
                all_site_id, all_site_label, 
                test_size=test_size, shuffle=True, random_state=random_state, stratify=all_site_label)

    return [([id_train, label_train], [id_test, label_test])]



def splitTrainTestKF(K_num=5, dataPath='../data/processed/augment/', sitenameList=['site2_ADNI3'], categoryDict={'CN': 0, 'MCI': 1, 'AD': 2}, random_state=0):
    r"""
    单或多中心数据混合, 划分K折训练集和测试集
    """

    all_site_id, all_site_label = getAllData(dataPath=dataPath, sitenameList=sitenameList, categoryDict=categoryDict)

    KF_datalist = []
    KF = StratifiedKFold(n_splits=K_num, shuffle=True, random_state=random_state)
    for train_index, val_index in KF.split(all_site_id, all_site_label):
        kf_train_id, kf_val_id = all_site_id[train_index], all_site_id[val_index]
        kf_train_label, kf_val_label = all_site_label[train_index], all_site_label[val_index]
        KF_datalist.append(([kf_train_id, kf_train_label], [kf_val_id, kf_val_label]))

    return KF_datalist # [(fold1), (fold2), ... ]



class DataSetBase(Dataset):
    def __init__(self, id, label, adj_type='pearson', adj_thr='pos', adj_norm=True, bold_pad=True):
        r'''
        adj_type: pearson, spearman
        adj_thr: 10, 20, ..., 30, 'pos'
        '''

        self.id_path = id # 被试文件夹
        self.label_numeric = label
        self.adj_type = adj_type
        self.adj_thr = adj_thr
        self.adj_norm = adj_norm
        self.bold_pad = bold_pad

    def __getitem__(self, index):
        subDirPath = self.id_path[index]
        label = self.label_numeric[index]

        bold = None
        adj_coef = None

        for subFileName in os.listdir(subDirPath): # 遍历被试文件夹

            if plat == 'windows':
                subFileName = subFileName.encode('utf-8').decode('utf-8')
            elif plat == 'linux':
                subFileName = subFileName.decode('utf-8')


            if self.bold_pad == True and 'bold_norm_padding.npy' == subFileName:
                bold = np.load(subDirPath + '/' + subFileName) # VT
            
            if self.bold_pad == False and 'bold_norm.npy' == subFileName:
                bold = np.load(subDirPath + '/' + subFileName) # VT

            if self.adj_type == 'pearson' and 'pearson_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)

            if self.adj_type == 'spearman' and 'spearman_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)

            if self.adj_type == 'cos' and 'cos_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)

            if self.adj_type == 'dcor' and 'dcor_mat_raw.npy' == subFileName:
                adj_coef = np.load(subDirPath + '/' + subFileName)


        if self.adj_thr != 'pos' and self.adj_norm == True: # GCN
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(vec_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr != 'pos' and self.adj_norm == False: # GAT
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(adj_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
        
        elif self.adj_thr == 'pos' and self.adj_norm == True: # GCN
            # if adj_coef.all()>=0.0:
            #     pass
            # else:
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr == 'pos' and self.adj_norm == False: # GAT
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I

        return subDirPath, bold, adj_coef, label # [V T], [V V], [1]


    def __len__(self):
        return len(self.label_numeric)






class DataSetRec(Dataset):
    def __init__(self, id, label, adj_type='pearson', adj_thr='pos', adj_norm=True, bold_pad=True):
        r'''
        adj_type: pearson, spearman
        adj_thr: 10, 20, ..., 30, 'pos'
        '''

        self.id_path = id # 被试文件夹列表
        self.label_numeric = label
        self.adj_type = adj_type
        self.adj_thr = adj_thr
        self.adj_norm = adj_norm
        self.bold_pad = bold_pad

        self.sampleTrans = []
        self.sampleMask = []
        self.sampleRec = []
        self.sampleAdj = []
        self.sampleLabel = []
        self.sampleSubId = []

        for i in range(len(self.id_path)):
            subDirPath = self.id_path[i]
            subLabel = self.label_numeric[i]
            sampleName = [] # 取出每个被试里的文件
            for subFileName in os.listdir(subDirPath):
                if plat == 'windows':
                    subFileName = subFileName.encode('utf-8').decode('utf-8')
                elif plat == 'linux':
                    subFileName = subFileName.decode('utf-8')
                if 'aug' in subFileName:
                    if subFileName.split('-')[0] not in sampleName:
                        sampleName.append(subFileName.split('-')[0]) # 扩充后的文件名，rec和mask成对同名，只取一个即可

            if self.bold_pad:
                for samplestr in sampleName:
                    self.sampleTrans.append(subDirPath + '/' + samplestr + '-trans.npy')
                    self.sampleMask.append(subDirPath + '/' + samplestr + '-mask.npy')
                    self.sampleRec.append(subDirPath + '/' + samplestr + '-rec.npy')
                    self.sampleLabel.append(subLabel)
                    self.sampleSubId.append(subDirPath + '/' + samplestr)
                    if adj_type=='pearson':
                        self.sampleAdj.append(subDirPath + '/' + 'pearson_mat_raw.npy')
                    

            # else:
            #     for samplestr in sampleName:
            #         self.sampleMask.append(subDirPath + '/' + samplestr + '-mask.npy')
            #         self.sampleRec.append(subDirPath + '/' + samplestr + '-rec.npy')
            #         self.sampleAdj.append(subDirPath + '/' + 'pear_mat_raw.npy')
            #         self.sampleLabel.append(subLabel)
            #         self.sampleSubId.append(subDirPath + '/' + samplestr)
                

    def __getitem__(self, index):
        trans = self.sampleTrans[index]
        mask = self.sampleMask[index]
        rec = self.sampleRec[index]
        adj_coef = self.sampleAdj[index]
        label = self.sampleLabel[index]
        id = self.sampleSubId[index]

        trans = np.load(trans)
        mask = np.load(mask)
        rec = np.load(rec)
        adj_coef = np.load(adj_coef)
        
        # trans = trans.T # TV -> VT
        # mask = mask.T
        # rec = rec.T # TV ->VT

        
        if self.adj_thr != 'pos' and self.adj_norm == True: # GCN
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(vec_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr != 'pos' and self.adj_norm == False: # GAT
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(adj_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I

        elif self.adj_thr == 'pos' and self.adj_norm == True: # GCN
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr == 'pos' and self.adj_norm == False: # GAT
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I

        return id, trans, mask, rec, adj_coef, label # [V T], [V T], [V V], [1,]


    def __len__(self):
        return len(self.sampleTrans)





class DataSetTrainAug(Dataset):
    def __init__(self, id, label, adj_type='pearson', adj_thr='pos', adj_norm=True, bold_pad=True):
        r"""
        adj_type: pearson, spearman
        adj_thr: 10, 20, ..., 30, 'pos'
        """

        self.id_path = id # 被试文件夹列表
        self.label_numeric = label

        self.adj_type = adj_type
        self.adj_thr = adj_thr
        self.adj_norm = adj_norm
        self.bold_pad = bold_pad

        self.sampleBold = []
        self.sampleAdj = []
        self.sampleLabel = []
        self.sampleSubId = []

        for i in range(len(self.id_path)):
            subDirPath = self.id_path[i]
            subLabel = self.label_numeric[i]

            # sampleName = [] # 取出每个被试里的文件
            
            for subFileName in os.listdir(subDirPath):
                if plat == 'windows':
                    subFileName = subFileName.encode('utf-8').decode('utf-8')
                elif plat == 'linux':
                    subFileName = subFileName.decode('utf-8')

                # if 'pad' in subFileName:
                    # sampleName.append(subFileName) # 扩充后的文件名，rec和mask成对同名，只取一个即可

            # if self.bold_pad:
            #     for samplestr in sampleName:
                    
                if 'pad' in subFileName:
                    self.sampleBold.append(subDirPath + '/' + subFileName)
                    self.sampleLabel.append(subLabel)
                    self.sampleSubId.append(subDirPath)
                    if adj_type=='pearson':
                        self.sampleAdj.append(subDirPath + '/' + 'pearson_mat_raw.npy')
                    
            # else:
            #     for samplestr in sampleName:
            #         self.sampleMask.append(subDirPath + '/' + samplestr + '-mask.npy')
            #         self.sampleRec.append(subDirPath + '/' + samplestr + '-rec.npy')
            #         self.sampleAdj.append(subDirPath + '/' + 'pear_mat_raw.npy')
            #         self.sampleLabel.append(subLabel)
            #         self.sampleSubId.append(subDirPath + '/' + samplestr)



    def __getitem__(self, index):
        bold = self.sampleBold[index]
        adj_coef = self.sampleAdj[index]
        label = self.sampleLabel[index]
        id = self.sampleSubId[index]

        bold = np.load(bold)
        adj_coef = np.load(adj_coef)
        
        # bold = bold.T # TV -> VT

        
        if self.adj_thr != 'pos' and self.adj_norm == True: # GCN
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(vec_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr != 'pos' and self.adj_norm == False: # GAT
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(adj_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I

        elif self.adj_thr == 'pos' and self.adj_norm == True: # GCN
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr == 'pos' and self.adj_norm == False: # GAT
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I

        return id, bold, adj_coef, label # [V T], [V V], [1,]


    def __len__(self):
        return len(self.sampleBold)






class DataSetTrainAugCross(Dataset):
    def __init__(self, src_id, src_label, dst_id, dst_label, adj_type='pearson', adj_thr='pos', adj_norm=True, bold_pad=True):
        r"""
        adj_type: pearson, spearman
        adj_thr: 10, 20, ..., 30, 'pos'
        """

        self.src_id_path = src_id # 被试文件夹列表
        self.src_label_numeric = src_label

        self.dst_id_path = dst_id # 被试文件夹列表
        self.dst_label_numeric = dst_label

        self.adj_type = adj_type
        self.adj_thr = adj_thr
        self.adj_norm = adj_norm
        self.bold_pad = bold_pad

        self.src_sampleBold = []
        self.src_sampleAdj = []
        self.src_sampleLabel = []
        self.src_sampleSubId = []

        self.dst_sampleBold = []
        self.dst_sampleAdj = []
        self.dst_sampleLabel = []
        self.dst_sampleSubId = []

        self.pair_sampleBold = []
        self.pair_sampleAdj = []
        self.pair_sampleLabel = []
        self.pair_sampleSubId = []


        for i in range(len(self.src_id_path)):
            subDirPath = self.src_id_path[i]
            subLabel = self.src_label_numeric[i]
            
            for subFileName in os.listdir(subDirPath):
                if plat == 'windows':
                    subFileName = subFileName.encode('utf-8').decode('utf-8')
                elif plat == 'linux':
                    subFileName = subFileName.decode('utf-8')

                if 'pad' in subFileName:
                    self.src_sampleBold.append(subDirPath + '/' + subFileName)
                    self.src_sampleLabel.append(subLabel)
                    self.src_sampleSubId.append(subDirPath)
                    if adj_type=='pearson':
                        self.src_sampleAdj.append(subDirPath + '/' + 'pearson_mat_raw.npy')



        for i in range(len(self.dst_id_path)):
            subDirPath = self.dst_id_path[i]
            subLabel = self.dst_label_numeric[i]
            
            for subFileName in os.listdir(subDirPath):
                if plat == 'windows':
                    subFileName = subFileName.encode('utf-8').decode('utf-8')
                elif plat == 'linux':
                    subFileName = subFileName.decode('utf-8')

                if 'pad' in subFileName:
                    self.dst_sampleBold.append(subDirPath + '/' + subFileName)
                    self.dst_sampleLabel.append(subLabel)
                    self.dst_sampleSubId.append(subDirPath)
                    if adj_type=='pearson':
                        self.dst_sampleAdj.append(subDirPath + '/' + 'pearson_mat_raw.npy')


        for src_ind in range(len(self.src_sampleBold)):
            # 对于每一个src，抽一个dst对应
            dst_list = [i for i in range(len(self.dst_sampleBold))]
            dst_ind = sample(dst_list, 1)[0]
            print('smapling dst ind: ', dst_ind)

            self.pair_sampleBold.append((self.src_sampleBold[src_ind], self.dst_sampleBold[dst_ind]))
            self.pair_sampleAdj.append((self.src_sampleAdj[src_ind], self.dst_sampleAdj[dst_ind]))
            self.pair_sampleLabel.append((self.src_sampleLabel[src_ind], self.dst_sampleLabel[dst_ind]))
            self.pair_sampleSubId.append((self.src_sampleSubId[src_ind], self.dst_sampleSubId[dst_ind]))




    def __getitem__(self, index):
        src_bold, dst_bold = self.pair_sampleBold[index]
        src_adj_coef, dst_adj_coef = self.pair_sampleAdj[index]
        src_label, dst_label = self.pair_sampleLabel[index]
        src_id, dst_id = self.pair_sampleSubId[index]

        src_bold = np.load(src_bold)
        dst_bold = np.load(dst_bold)

        src_adj_coef = np.load(src_adj_coef)
        dst_adj_coef = np.load(dst_adj_coef)

        # src_bold = src_bold.T # TV -> VT
        # dst_bold = dst_bold.T

        if self.adj_thr == 'pos' and self.adj_norm == True: # GCN
            src_adj_coef[src_adj_coef<0.0] = 0.0
            I = np.eye(src_adj_coef.shape[0])
            src_adj_coef = src_adj_coef + I
            d = src_adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            src_adj_coef = np.matmul(np.matmul(D, src_adj_coef), D)


            dst_adj_coef[dst_adj_coef<0.0] = 0.0
            I = np.eye(dst_adj_coef.shape[0])
            dst_adj_coef = dst_adj_coef + I
            d = dst_adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            dst_adj_coef = np.matmul(np.matmul(D, dst_adj_coef), D)


        return src_id, dst_id, src_bold, dst_bold, src_adj_coef, dst_adj_coef, src_label, dst_label # [V T], [V V], [1,]


    def __len__(self):
        return len(self.pair_sampleBold)








class DataSetTrainBalance(Dataset):
    def __init__(self, id, label, adj_type='pearson', adj_thr='pos', adj_norm=True, bold_pad=True):
        r"""
        adj_type: pearson, spearman
        adj_thr: 10, 20, ..., 30, 'pos'
        """

        self.id_path = id # 被试文件夹列表
        self.label_numeric = label

        self.adj_type = adj_type
        self.adj_thr = adj_thr
        self.adj_norm = adj_norm
        self.bold_pad = bold_pad

        self.sampleTrans = []
        self.sampleMask = []
        self.sampleRec = []
        self.sampleAdj = []
        self.sampleLabel = []
        self.sampleSubId = []

        for i in range(len(self.id_path)):
            subDirPath = self.id_path[i]
            subLabel = self.label_numeric[i]
            sampleName = [] # 取出每个被试里的文件
            for subFileName in os.listdir(subDirPath):
                if plat == 'windows':
                    subFileName = subFileName.encode('utf-8').decode('utf-8')
                elif plat == 'linux':
                    subFileName = subFileName.decode('utf-8')
                if 'aug' in subFileName:
                    if subFileName.split('-')[0] not in sampleName:
                        sampleName.append(subFileName.split('-')[0]) # 扩充后的文件名，rec和mask成对同名，只取一个即可

            if self.bold_pad:
                for samplestr in sampleName:
                    self.sampleTrans.append(subDirPath + '/' + samplestr + '-trans.npy')
                    self.sampleMask.append(subDirPath + '/' + samplestr + '-mask.npy')
                    self.sampleRec.append(subDirPath + '/' + samplestr + '-rec.npy')
                    self.sampleLabel.append(subLabel)
                    self.sampleSubId.append(subDirPath + '/' + samplestr)
                    if adj_type=='pearson':
                        self.sampleAdj.append(subDirPath + '/' + 'pearson_mat_raw.npy')
                    elif adj_type=='spearman':
                        self.sampleAdj.append(subDirPath + '/' + 'spearman_mat_raw.npy')
                    elif adj_type=='cos':
                        self.sampleAdj.append(subDirPath + '/' + 'cos_mat_raw.npy')
                    elif adj_type=='dcor':
                        self.sampleAdj.append(subDirPath + '/' + 'dcor_mat_raw.npy')

            # else:
            #     for samplestr in sampleName:
            #         self.sampleMask.append(subDirPath + '/' + samplestr + '-mask.npy')
            #         self.sampleRec.append(subDirPath + '/' + samplestr + '-rec.npy')
            #         self.sampleAdj.append(subDirPath + '/' + 'pear_mat_raw.npy')
            #         self.sampleLabel.append(subLabel)
            #         self.sampleSubId.append(subDirPath + '/' + samplestr)

        # print()
        # print('sampleMask: ', self.sampleMask)
        # print('sampleRec: ', self.sampleRec)
        # print('sampleAdj: ', self.sampleAdj)
        # print('sampleLabel: ', self.sampleLabel)
        # print('sampleSubId: ', self.sampleSubId)


        self.negSampleTrans = []
        self.negSampleMask = []
        self.negSampleRec = []
        self.negSampleAdj = []
        self.negSampleLabel = []
        self.negSampleSubId = []

        self.posSampleTrans = []
        self.posSampleMask = []
        self.posSampleRec = []
        self.posSampleAdj = []
        self.posSampleLabel = []
        self.posSampleSubId = []
        
        self.sampleTransBalance = []
        self.sampleMaskBalance = []
        self.sampleRecBalance = []
        self.sampleAdjBalance = []
        self.sampleLabelBalance = []
        self.sampleSubIdBalance = []

        for i in range(len(self.sampleLabel)):
            if self.sampleLabel[i] == 0:
                self.negSampleTrans.append(self.sampleTrans[i])
                self.negSampleMask.append(self.sampleMask[i])
                self.negSampleRec.append(self.sampleRec[i])
                self.negSampleAdj.append(self.sampleAdj[i])
                self.negSampleLabel.append(self.sampleLabel[i])
                self.negSampleSubId.append(self.sampleSubId[i])
            
            elif self.sampleLabel[i] == 1:
                self.posSampleTrans.append(self.sampleTrans[i])
                self.posSampleMask.append(self.sampleMask[i])
                self.posSampleRec.append(self.sampleRec[i])
                self.posSampleAdj.append(self.sampleAdj[i])
                self.posSampleLabel.append(self.sampleLabel[i])
                self.posSampleSubId.append(self.sampleSubId[i])
        
        neg_merge = list(zip(self.negSampleTrans, self.negSampleMask, self.negSampleRec, self.negSampleAdj, self.negSampleLabel, self.negSampleSubId))
        pos_merge = list(zip(self.posSampleTrans, self.posSampleMask, self.posSampleRec, self.posSampleAdj, self.posSampleLabel, self.posSampleSubId))
        random.shuffle(neg_merge)
        random.shuffle(pos_merge)

        if len(neg_merge) > len(pos_merge):
            neg_merge = neg_merge[0:len(pos_merge)]
            pos_merge = pos_merge[:]
        else:
            neg_merge = neg_merge[:]
            pos_merge = pos_merge[0:len(neg_merge)]
        
        self.negSampleTrans[:], self.negSampleMask[:], self.negSampleRec[:], self.negSampleAdj[:], self.negSampleLabel[:], self.negSampleSubId[:] = zip(*neg_merge)
        self.posSampleTrans[:], self.posSampleMask[:], self.posSampleRec[:], self.posSampleAdj[:], self.posSampleLabel[:], self.posSampleSubId[:] = zip(*pos_merge)

        self.sampleTransBalance = self.negSampleTrans+self.posSampleTrans
        self.sampleMaskBalance = self.negSampleMask+self.posSampleMask
        self.sampleRecBalance = self.negSampleRec+self.posSampleRec
        self.sampleAdjBalance = self.negSampleAdj+self.posSampleAdj
        self.sampleLabelBalance = self.negSampleLabel+self.posSampleLabel
        self.sampleSubIdBalance = self.negSampleSubId+self.posSampleSubId



    def __getitem__(self, index):
        trans = self.sampleTransBalance[index]
        mask = self.sampleMaskBalance[index]
        rec = self.sampleRecBalance[index]
        adj_coef = self.sampleAdjBalance[index]
        label = self.sampleLabelBalance[index]
        id = self.sampleSubIdBalance[index]

        trans = np.load(trans)
        mask = np.load(mask)
        rec = np.load(rec)
        adj_coef = np.load(adj_coef)
        
        # trans = trans.T # TV -> VT
        # mask = mask.T
        # rec = rec.T # TV ->VT

        
        if self.adj_thr != 'pos' and self.adj_norm == True: # GCN
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(vec_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr != 'pos' and self.adj_norm == False: # GAT
            vec_coef = adj_coef[np.triu_indices(n=90, k=1)] # 取出上三角元素，对角线除外
            thr_value = np.percentile(adj_coef, 100-self.adj_thr)
            adj_coef[adj_coef<thr_value] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I

        elif self.adj_thr == 'pos' and self.adj_norm == True: # GCN
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I
            d = adj_coef.sum(1)
            D = np.diag(np.power(d, -0.5))
            adj_coef = np.matmul(np.matmul(D, adj_coef), D)

        elif self.adj_thr == 'pos' and self.adj_norm == False: # GAT
            adj_coef[adj_coef<0.0] = 0.0
            I = np.eye(adj_coef.shape[0])
            adj_coef = adj_coef + I

        return id, trans, mask, rec, adj_coef, label # [V T], [V T], [V V], [1,]


    def __len__(self):
        return len(self.sampleTransBalance)

