'''
Z-Score 归一化
'''

import numpy as np
import os


def boldNorm(src_path='../data/processed/source/site1_ADNI2', dst_path='../data/processed/source/site1_ADNI2'):

    for subDir in os.listdir(src_path): # 遍历站点内的文件夹
        print('subDirPath: ', src_path + '/' + subDir)

        bold = np.load(src_path + '/' + subDir + '/' + 'bold.npy') # VT
        bold_norm = (bold - np.mean(bold)) / np.std(bold)

        np.save(dst_path + '/' + subDir + '/' + 'bold_norm.npy', bold_norm.astype(np.float32))


site1_ADNI2 = '../data/processed/source/site1_ADNI2'
site2_ADNI3 = '../data/processed/source/site2_ADNI3'
site3_Xuanwu1 = '../data/processed/source/site3_Xuanwu1'
site4_Xuanwu2 = '../data/processed/source/site4_Xuanwu2'
site5_Tongji1 = '../data/processed/source/site5_Tongji1'
site6_Tongji2 = '../data/processed/source/site6_Tongji2'


if __name__ == '__main__':

    boldNorm(src_path=site1_ADNI2, dst_path=site1_ADNI2)
    boldNorm(src_path=site2_ADNI3, dst_path=site2_ADNI3)
    boldNorm(src_path=site3_Xuanwu1, dst_path=site3_Xuanwu1)
    boldNorm(src_path=site4_Xuanwu2, dst_path=site4_Xuanwu2)
    boldNorm(src_path=site5_Tongji1, dst_path=site5_Tongji1)
    boldNorm(src_path=site6_Tongji2, dst_path=site6_Tongji2)
