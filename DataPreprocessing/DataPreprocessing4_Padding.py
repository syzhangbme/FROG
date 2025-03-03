import numpy as np
import os
import math


def Padding(X, maxLen=230):
    r'''
    input: [V T] 归一化 或 归一化mask 或 归一化rec 的BOLD信号
    function: 在BOLD信号后面用padVal值填充至最大长度maxLen, 或截断超出的长度
    output: [V T] 填充后的 归一化 或 归一化mask 或 归一化rec 的BOLD信号
    '''
    Xc = X.copy()
    seqLen = Xc.shape[1]
    if seqLen < maxLen:
        
        X_pad = np.concatenate((np.tile(Xc, (1, math.floor(maxLen/seqLen))), Xc[:, 0:maxLen-seqLen*math.floor(maxLen/seqLen)]), axis=1)
        return X_pad

    elif seqLen == maxLen: # 保留
        return Xc

    else:
        return Xc[:, 0:maxLen] # 截断


def boldPad(src_path='../data/processed/site1_ADNI2', dst_path='../data/processed/site1_ADNI2'):

    # 找出最大长度
    seq_len = []
    for subDir in os.listdir(src_path): # 遍历站点内的文件夹
        bold = np.load(src_path + '/' + subDir + '/' + 'bold_norm.npy') # [V T]
        seq_len.append(bold.shape[1])

    # maxLen = max(seq_len)
    maxLen = 230
    for subDir in os.listdir(src_path): # 遍历站点内的文件夹
        print('subDirPath: ', src_path + '/' + subDir)
    
        bold = np.load(src_path + '/' + subDir + '/' + 'bold_norm.npy') # [V T]
        bold_pad = Padding(bold, maxLen=maxLen)
    
        if not os.path.exists(dst_path + '/' +subDir): # 创建写入目标的被试文件夹
            os.makedirs(dst_path + '/' +subDir) 

        np.save(dst_path + '/' +subDir + '/' + 'bold_norm_padding.npy', bold_pad.astype(np.float32))


site1_ADNI2 = '../data/processed/source/site1_ADNI2'
site2_ADNI3 = '../data/processed/source/site2_ADNI3'
site3_Xuanwu1 = '../data/processed/source/site3_Xuanwu1'
site4_Xuanwu2 = '../data/processed/source/site4_Xuanwu2'
site5_Tongji1 = '../data/processed/source/site5_Tongji1'
site6_Tongji2 = '../data/processed/source/site6_Tongji2'


if __name__ == '__main__':

    boldPad(src_path=site1_ADNI2, dst_path=site1_ADNI2)
    boldPad(src_path=site2_ADNI3, dst_path=site2_ADNI3)
    boldPad(src_path=site3_Xuanwu1, dst_path=site3_Xuanwu1)
    boldPad(src_path=site4_Xuanwu2, dst_path=site4_Xuanwu2)
    boldPad(src_path=site5_Tongji1, dst_path=site5_Tongji1)
    boldPad(src_path=site6_Tongji2, dst_path=site6_Tongji2)
