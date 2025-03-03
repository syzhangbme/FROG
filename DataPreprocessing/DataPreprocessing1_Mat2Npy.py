"""
matlab处理后的数据 再用python处理
此代码将样本格式从mat转为npy
"""

import os
import shutil
import scipy.io as scio
import numpy as np


# CN-0, SCD-1, MCI-2, AD-3
def ADmat2npy(src_path, dst_path, category):
    r"""
    src_path: string, 原始路径
    dst_path: string, 目标路径
    category: string, 原始路径下样本的类别
    """
    for filename in os.listdir(src_path):
        bold = scio.loadmat(src_path + filename)['imgDataNode'].T # TV -> VT
        # print(bold)
        label = 'NULL'
        if category == 'CN':
            label = 'CN'
        elif category == 'SCD':
            label = 'SCD'
        elif category == 'MCI':
            label = 'MCI'
        elif category == 'AD':
            label = 'AD'

        # 如果存在被试子目录，则删除，重新创建
        dst_path_sample = dst_path+filename[:-4]
        if os.path.exists(dst_path_sample):
            shutil.rmtree(dst_path_sample)
        os.mkdir(dst_path_sample)
        print(dst_path_sample)
        np.save(dst_path_sample + '/bold.npy', np.array(bold).astype(np.float32))
        with open(dst_path_sample + '/label.txt', 'w') as f:
            f.write(label)



#----------------------------------- ADNI2-------------------------------------
site1_ADNI2_CN_path_src = '../data/raw/site1_ADNI2_130/CN_139/'
site1_ADNI2_MCI_path_src = '../data/raw/site1_ADNI2_130/MCI_244/'
site1_ADNI2_AD_path_src = '../data/raw/site1_ADNI2_130/AD_132/'

site1_ADNI2_path_dst = '../data/processed/source/site1_ADNI2/'


#----------------------------------- ADNI3-------------------------------------
site2_ADNI3_CN_path_src1 = '../data/raw/site2_ADNI3_187/DICOM_number_197/NC_305/'
site2_ADNI3_SCD_path_src1 = '../data/raw/site2_ADNI3_187/DICOM_number_197/SMC_39/'
site2_ADNI3_MCI_path_src1 = '../data/raw/site2_ADNI3_187/DICOM_number_197/MCI_149/'
site2_ADNI3_AD_path_src1 = '../data/raw/site2_ADNI3_187/DICOM_number_197/AD_45/'

site2_ADNI3_CN_path_src2 = '../data/raw/site2_ADNI3_187/DICOM_number_9600/NC_135/'
site2_ADNI3_SCD_path_src2 = '../data/raw/site2_ADNI3_187/DICOM_number_9600/SMC_52/'
site2_ADNI3_MCI_path_src2 = '../data/raw/site2_ADNI3_187/DICOM_number_9600/MCI_83/'
site2_ADNI3_AD_path_src2 = '../data/raw/site2_ADNI3_187/DICOM_number_9600/AD_45/'

site2_ADNI3_path_dst = '../data/processed/source/site2_ADNI3/'


#----------------------------------- Xuanwu1-------------------------------------
site3_Xuanwu1_CN_path_src = '../data/raw/site3_Xuanwu1_229/NC_148/'
site3_Xuanwu1_SCD_path_src = '../data/raw/site3_Xuanwu1_229/SCD_72/'
site3_Xuanwu1_MCI_path_src = '../data/raw/site3_Xuanwu1_229/MCI_121/'
site3_Xuanwu1_AD_path_src = '../data/raw/site3_Xuanwu1_229/AD_38/'

site3_Xuanwu1_path_dst = '../data/processed/source/site3_Xuanwu1/'


# ----------------------------------- Xuanwu2-------------------------------------
site4_Xuanwu2_CN_path_src = '../data/raw/site4_Xuanwu2_230/NC_241/'
site4_Xuanwu2_SCD_path_src = '../data/raw/site4_Xuanwu2_230/SCD_230/'
site4_Xuanwu2_MCI_path_src = '../data/raw/site4_Xuanwu2_230/MCI_46/'
site4_Xuanwu2_AD_path_src = '../data/raw/site4_Xuanwu2_230/AD_18/'

site4_Xuanwu2_path_dst = '../data/processed/source/site4_Xuanwu2/'


#----------------------------------- Tongji1-------------------------------------
site5_Tongji1_CN_path_src = '../data/raw/site5_Tongji1_230/NC_71/'
site5_Tongji1_MCI_path_src = '../data/raw/site5_Tongji1_230/MCI_85/'
site5_Tongji1_AD_path_src = '../data/raw/site5_Tongji1_230/AD_57/'

site5_Tongji1_path_dst = '../data/processed/source/site5_Tongji1/'


#----------------------------------- Tongji2-------------------------------------
site6_Tongji2_CN_path_src = '../data/raw/site6_Tongji2_950/NC_66/'
site6_Tongji2_MCI_path_src = '../data/raw/site6_Tongji2_950/MCI_62/'
site6_Tongji2_AD_path_src = '../data/raw/site6_Tongji2_950/AD_42/'

site6_Tongji2_path_dst = '../data/processed/source/site6_Tongji2/'



if __name__ == '__main__':
    os.makedirs('../data/processed/source/site1_ADNI2/')
    os.makedirs('../data/processed/source/site2_ADNI3/')
    os.makedirs('../data/processed/source/site3_Xuanwu1/')
    os.makedirs('../data/processed/source/site4_Xuanwu2/')
    os.makedirs('../data/processed/source/site5_Tongji1/')
    os.makedirs('../data/processed/source/site6_Tongji2/')

    ADmat2npy(site1_ADNI2_CN_path_src, site1_ADNI2_path_dst, 'CN')
    ADmat2npy(site1_ADNI2_MCI_path_src, site1_ADNI2_path_dst, 'MCI')
    ADmat2npy(site1_ADNI2_AD_path_src, site1_ADNI2_path_dst, 'AD')

    ADmat2npy(site2_ADNI3_CN_path_src1, site2_ADNI3_path_dst, 'CN')
    ADmat2npy(site2_ADNI3_SCD_path_src1, site2_ADNI3_path_dst, 'SCD')
    ADmat2npy(site2_ADNI3_MCI_path_src1, site2_ADNI3_path_dst, 'MCI')
    ADmat2npy(site2_ADNI3_AD_path_src1, site2_ADNI3_path_dst, 'AD')
    ADmat2npy(site2_ADNI3_CN_path_src2, site2_ADNI3_path_dst, 'CN')
    ADmat2npy(site2_ADNI3_SCD_path_src2, site2_ADNI3_path_dst, 'SCD')
    ADmat2npy(site2_ADNI3_MCI_path_src2, site2_ADNI3_path_dst, 'MCI')
    ADmat2npy(site2_ADNI3_AD_path_src2, site2_ADNI3_path_dst, 'AD')

    ADmat2npy(site3_Xuanwu1_CN_path_src, site3_Xuanwu1_path_dst, 'CN')
    ADmat2npy(site3_Xuanwu1_SCD_path_src, site3_Xuanwu1_path_dst, 'SCD')
    ADmat2npy(site3_Xuanwu1_MCI_path_src, site3_Xuanwu1_path_dst, 'MCI')
    ADmat2npy(site3_Xuanwu1_AD_path_src, site3_Xuanwu1_path_dst, 'AD')

    ADmat2npy(site4_Xuanwu2_CN_path_src, site4_Xuanwu2_path_dst, 'CN')
    ADmat2npy(site4_Xuanwu2_SCD_path_src, site4_Xuanwu2_path_dst, 'SCD')
    ADmat2npy(site4_Xuanwu2_MCI_path_src, site4_Xuanwu2_path_dst, 'MCI')
    ADmat2npy(site4_Xuanwu2_AD_path_src, site4_Xuanwu2_path_dst, 'AD')

    ADmat2npy(site5_Tongji1_CN_path_src, site5_Tongji1_path_dst, 'CN')
    ADmat2npy(site5_Tongji1_MCI_path_src, site5_Tongji1_path_dst, 'MCI')
    ADmat2npy(site5_Tongji1_AD_path_src, site5_Tongji1_path_dst, 'AD')

    ADmat2npy(site6_Tongji2_CN_path_src, site6_Tongji2_path_dst, 'CN')
    ADmat2npy(site6_Tongji2_MCI_path_src, site6_Tongji2_path_dst, 'MCI')
    ADmat2npy(site6_Tongji2_AD_path_src, site6_Tongji2_path_dst, 'AD')
    