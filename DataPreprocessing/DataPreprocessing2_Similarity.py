import os
import numpy as np
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# 计算皮尔逊相关系数
def pear_coef(matrix):
    r'''
    功能: 计算皮尔逊相关性
    输入: 二维矩阵(节点数量，序列长度), 即单个样本的BOLD信号
    输出: 单个样本的 皮尔逊原始系数矩阵, 即对角元素是0, 其他位置是对称分布的皮尔逊相关系数
    '''
    pear_mat = np.zeros((90, 90))
    for i in range(0, matrix.shape[0]-1): # 节点数量
        for j in range(i+1, matrix.shape[0]):
            coer = pearsonr(matrix[i, :], matrix[j, :])[0]
            if math.isinf(coer): # 异常值处理
                coer = 0
            if math.isnan(coer):
                coer = 0
            pear_mat[i, j] = coer
            pear_mat[j, i] = coer
    return pear_mat




def buildSim(src_path='../data/processed/source/site1_ADNI2', dst_path='../data/processed/source/site1_ADNI2'):
    for subDir in os.listdir(src_path): # 遍历站点内的文件夹
        print('subDirPath: ', src_path + '/' + subDir)

        bold = np.load(src_path + '/' + subDir + '/' + 'bold.npy')
        pear_mat = pear_coef(bold)
        # 保存不取绝对值不卡阈值的原始Pearson相关性系数矩阵, 对角元素是0
        np.save(dst_path + '/' + subDir + '/' + 'pearson_mat_raw.npy', pear_mat.astype(np.float32))



site1_ADNI2 = '../data/processed/source/site1_ADNI2'
site2_ADNI3 = '../data/processed/source/site2_ADNI3'
site3_Xuanwu1 = '../data/processed/source/site3_Xuanwu1'
site4_Xuanwu2 = '../data/processed/source/site4_Xuanwu2'
site5_Tongji1 = '../data/processed/source/site5_Tongji1'
site6_Tongji2 = '../data/processed/source/site6_Tongji2'


if __name__ == '__main__':
    buildSim(src_path=site1_ADNI2, dst_path=site1_ADNI2)
    buildSim(src_path=site2_ADNI3, dst_path=site2_ADNI3)
    buildSim(src_path=site3_Xuanwu1, dst_path=site3_Xuanwu1)
    buildSim(src_path=site4_Xuanwu2, dst_path=site4_Xuanwu2)
    buildSim(src_path=site5_Tongji1, dst_path=site5_Tongji1)
    buildSim(src_path=site6_Tongji2, dst_path=site6_Tongji2)
