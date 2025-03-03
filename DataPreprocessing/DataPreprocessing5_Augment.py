import numpy as np
import os
import math
import shutil


def maskNode(X, maskV=20, maskVal=0.0):
    r'''
    input: [V T] 归一化后的BOLD信号
    function: 随机选取maskV个脑区的BOLD信号, 用maskVal值掩盖
    output: [V T] 掩膜处理后的BOLD信号, 作为重建的输入; [V T] 被掩膜覆盖之前的部分, 作为重建的标签
    '''
    Xc = X.copy()
    mask = np.zeros_like(Xc)
    V_num = Xc.shape[0]
    index_cand = [i for i in range(V_num)]
    index_npy = np.random.choice(index_cand, maskV, replace=False)
    index_list = index_npy.tolist()
    index_list = sorted(index_list)
    mask[index_list, :] = 1
    y = Xc * mask
    Xc[index_list, :] = maskVal
    return Xc, y, mask

def maskTime(X, maskT=30, maskVal=0.0):
    r'''
    input: [V T] 归一化后的BOLD信号
    function: 随机选取maskT时间跨度的BOLD信号, 用maskVal值掩盖
    output: [V T] 掩膜处理后的BOLD信号, 作为重建的输入; [V T] 被掩膜覆盖之前的部分, 作为重建的标签
    '''
    Xc = X.copy()
    mask = np.zeros_like(Xc)
    T_len = Xc.shape[1]
    index_cand = [i for i in range(T_len-maskT)]
    index_selected = np.random.choice(index_cand, 1, replace=False)[0]
    mask_index_start = int(index_selected)
    mask_index_end = int(mask_index_start + maskT)
    mask[:, mask_index_start:mask_index_end] = 1
    y = Xc * mask
    Xc[:, mask_index_start:mask_index_end] = maskVal
    return Xc, y, mask




def boldAugment(src_path='../data/processed/source/site1_ADNI2', dst_path='../data/processed/augment/site1_ADNI2'):

    for subDir in os.listdir(src_path): # 遍历站点内的文件夹
        print('copy: ', subDir)
        shutil.copytree(src_path + '/' +subDir, dst_path + '/' +subDir)

    for subDir in os.listdir(src_path): # 遍历站点内的文件夹
        print('subDirPath: ', src_path + '/' + subDir)

        bold_norm = np.load(src_path + '/' + subDir + '/' + 'bold_norm_padding.npy') # [V T]
        
        for j in range(2):
            bold_norm_T_trans, bold_norm_T_rec, bold_norm_T_mask = maskTime(bold_norm, maskT=30, maskVal=0.0) # 时间增强

            np.save(dst_path + '/' + subDir + '/' + 'bold_norm_pad_aug_T_{:s}-trans.npy'.format(str(j+1)), 
                    bold_norm_T_trans.astype(np.float32))
            np.save(dst_path + '/' + subDir + '/' + 'bold_norm_pad_aug_T_{:s}-rec.npy'.format(str(j+1)), 
                    bold_norm_T_rec.astype(np.float32))
            np.save(dst_path + '/' + subDir + '/' + 'bold_norm_pad_aug_T_{:s}-mask.npy'.format(str(j+1)), 
                    bold_norm_T_mask.astype(np.int32))
            
            
            bold_norm_V_trans, bold_norm_V_rec, bold_norm_V_mask = maskNode(bold_norm, maskV=20, maskVal=0.0) # 空间增强

            np.save(dst_path + '/' + subDir + '/' + 'bold_norm_pad_aug_V_{:s}-trans.npy'.format(str(j+1)), 
                    bold_norm_V_trans.astype(np.float32))
            np.save(dst_path + '/' + subDir + '/' + 'bold_norm_pad_aug_V_{:s}-rec.npy'.format(str(j+1)), 
                    bold_norm_V_rec.astype(np.float32))
            np.save(dst_path + '/' + subDir + '/' + 'bold_norm_pad_aug_V_{:s}-mask.npy'.format(str(j+1)), 
                    bold_norm_V_mask.astype(np.int32))


src_site1_ADNI2 = '../data/processed/source/site1_ADNI2'
src_site2_ADNI3 = '../data/processed/source/site2_ADNI3'
src_site3_Xuanwu1 = '../data/processed/source/site3_Xuanwu1'
src_site4_Xuanwu2 = '../data/processed/source/site4_Xuanwu2'
src_site5_Tongji1 = '../data/processed/source/site5_Tongji1'
src_site6_Tongji2 = '../data/processed/source/site6_Tongji2'

aug_site1_ADNI2 = '../data/processed/augment/site1_ADNI2'
aug_site2_ADNI3 = '../data/processed/augment/site2_ADNI3'
aug_site3_Xuanwu1 = '../data/processed/augment/site3_Xuanwu1'
aug_site4_Xuanwu2 = '../data/processed/augment/site4_Xuanwu2'
aug_site5_Tongji1 = '../data/processed/augment/site5_Tongji1'
aug_site6_Tongji2 = '../data/processed/augment/site6_Tongji2'



if __name__ == '__main__':
    os.makedirs(aug_site1_ADNI2)
    os.makedirs(aug_site2_ADNI3)
    os.makedirs(aug_site3_Xuanwu1)
    os.makedirs(aug_site4_Xuanwu2)
    os.makedirs(aug_site5_Tongji1)
    os.makedirs(aug_site6_Tongji2)

    boldAugment(src_path=src_site1_ADNI2, dst_path=aug_site1_ADNI2)
    boldAugment(src_path=src_site2_ADNI3, dst_path=aug_site2_ADNI3)
    boldAugment(src_path=src_site3_Xuanwu1, dst_path=aug_site3_Xuanwu1)
    boldAugment(src_path=src_site4_Xuanwu2, dst_path=aug_site4_Xuanwu2)
    boldAugment(src_path=src_site5_Tongji1, dst_path=aug_site5_Tongji1)
    boldAugment(src_path=src_site6_Tongji2, dst_path=aug_site6_Tongji2)
