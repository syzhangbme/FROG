import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



ROI_name = [
    'PreCG.L', 'PreCG.R', 'SFGdor.L', 'SFGdor.R', 'ORBsup.L', 'ORBsup.R', 'MFG.L', 'MFG.R', 'ORBmid.L', 'ORBmid.R',
    'IFGoperc.L', 'IFGoperc.R', 'IFGtriang.L', 'IFGtriang.R', 'ORBinf.L', 'ORBinf.R', 'ROL.L', 'ROL.R', 'SMA.L', 'SMA.R',
    'OLF.L', 'OLF.R', 'SFGmed.L', 'SFGmed.R', 'ORBsupmed.L', 'ORBsupmed.R', 'REC.L', 'REC.R', 'INS.L', 'INS.R',
    'ACG.L', 'ACG.R', 'DCG.L', 'DCG.R', 'PCG.L', 'PCG.R', 'HIP.L', 'HIP.R', 'PHG.L', 'PHG.R',
    'AMYG.L', 'AMYG.R', 'CAL.L', 'CAL.R', 'CUN.L', 'CUN.R', 'LING.L', 'LING.R', 'SOG.L', 'SOG.R',
    'MOG.L', 'MOG.R', 'IOG.L', 'IOG.R', 'FFG.L', 'FFG.R', 'PoCG.L', 'PoCG.R', 'SPG.L', 'SPG.R',
    'IPL.L', 'IPL.R', 'SMG.L', 'SMG.R', 'ANG.L', 'ANG.R', 'PCUN.L', 'PCUN.R', 'PCL.L', 'PCL.R',
    'CAU.L', 'CAU.R', 'PUT.L', 'PUT.R', 'PAL.L', 'PAL.R', 'THA.L', 'THA.R', 'HES.L', 'HES.R',
    'STG.L', 'STG.R', 'TPOsup.L', 'TPOsup.R', 'MTG.L', 'MTG.R', 'TPOmid.L', 'TPOmid.R', 'ITG.L', 'ITG.R'
    ]

subnetwork_category = {
    'Sensorimotor': 1,
    'Default mode': 2,
    'Attention': 3,
    'Visual': 4,
    'Subcortical': 5
    }


ROI2subnetwork = {
    'PreCG.L': 'Sensorimotor', 'PreCG.R': 'Sensorimotor', 'SFGdor.L': 'Default mode', 'SFGdor.R': 'Default mode', 
    'ORBsup.L': 'Attention', 'ORBsup.R': 'Default mode', 'MFG.L': 'Attention', 'MFG.R': 'Attention', 
    'ORBmid.L': 'Attention', 'ORBmid.R': 'Attention','IFGoperc.L': 'Attention', 'IFGoperc.R': 'Attention', 
    'IFGtriang.L': 'Attention', 'IFGtriang.R': 'Attention', 'ORBinf.L': 'Attention', 'ORBinf.R': 'Attention', 
    'ROL.L': 'Sensorimotor', 'ROL.R': 'Sensorimotor', 'SMA.L': 'Attention', 'SMA.R': 'Sensorimotor',
    'OLF.L': 'Subcortical', 'OLF.R': 'Subcortical', 'SFGmed.L': 'Default mode', 'SFGmed.R': 'Default mode', 
    'ORBsupmed.L': 'Default mode', 'ORBsupmed.R': 'Default mode', 'REC.L': 'Default mode', 'REC.R': 'Default mode', 
    'INS.L': 'Sensorimotor', 'INS.R': 'Sensorimotor', 'ACG.L': 'Default mode', 'ACG.R': 'Default mode', 
    'DCG.L': 'Subcortical', 'DCG.R': 'Subcortical', 'PCG.L': 'Default mode', 'PCG.R': 'Default mode', 
    'HIP.L': 'Subcortical', 'HIP.R': 'Subcortical', 'PHG.L': 'Subcortical', 'PHG.R': 'Subcortical',
    'AMYG.L': 'Subcortical', 'AMYG.R': 'Subcortical', 'CAL.L': 'Visual', 'CAL.R': 'Visual', 
    'CUN.L': 'Visual', 'CUN.R': 'Visual', 'LING.L': 'Visual', 'LING.R': 'Visual', 
    'SOG.L': 'Visual', 'SOG.R': 'Visual', 'MOG.L': 'Visual', 'MOG.R': 'Visual', 
    'IOG.L': 'Visual', 'IOG.R': 'Visual', 'FFG.L': 'Visual', 'FFG.R': 'Visual', 
    'PoCG.L': 'Sensorimotor', 'PoCG.R': 'Sensorimotor', 'SPG.L': 'Sensorimotor', 'SPG.R': 'Sensorimotor',
    'IPL.L': 'Attention', 'IPL.R': 'Attention', 'SMG.L': 'Sensorimotor', 'SMG.R': 'Sensorimotor', 
    'ANG.L': 'Attention', 'ANG.R': 'Attention', 'PCUN.L': 'Default mode', 'PCUN.R': 'Default mode', 
    'PCL.L': 'Sensorimotor', 'PCL.R': 'Sensorimotor', 'CAU.L': 'Subcortical', 'CAU.R': 'Subcortical', 
    'PUT.L': 'Subcortical', 'PUT.R': 'Subcortical', 'PAL.L': 'Subcortical', 'PAL.R': 'Subcortical', 
    'THA.L': 'Subcortical', 'THA.R': 'Subcortical', 'HES.L': 'Sensorimotor', 'HES.R': 'Sensorimotor',
    'STG.L': 'Sensorimotor', 'STG.R': 'Sensorimotor', 'TPOsup.L': 'Attention', 'TPOsup.R': 'Sensorimotor', 
    'MTG.L': 'Default mode', 'MTG.R': 'Default mode', 'TPOmid.L': 'Subcortical', 'TPOmid.R': 'Subcortical', 
    'ITG.L': 'Attention', 'ITG.R': 'Default mode',
    }



subnetwork2ROI = {
    'Sensorimotor' : ['PreCG.L', 'PreCG.R', 'ROL.L', 'ROL.R',  'SMA.R', 'INS.L', 'INS.R', 
    'PoCG.L', 'PoCG.R', 'SPG.L', 'SPG.R',  'SMG.L', 'SMG.R', 
    'PCL.L', 'PCL.R', 'HES.L', 'HES.R', 'STG.L', 'STG.R', 'TPOsup.R'], # 20 I


    'Default mode' : ['SFGdor.L', 'SFGdor.R', 'ORBsup.R', 'SFGmed.L', 'SFGmed.R', 
    'ORBsupmed.L', 'ORBsupmed.R', 'REC.L', 'REC.R', 
    'ACG.L', 'ACG.R', 'PCG.L', 'PCG.R', 'PCUN.L', 'PCUN.R', 
    'MTG.L', 'MTG.R', 'ITG.R'], # 18 IV


    'Attention' : ['ORBsup.L', 'MFG.L', 'MFG.R', 'ORBmid.L', 'ORBmid.R', 'IFGoperc.L', 'IFGoperc.R', 
    'IFGtriang.L', 'IFGtriang.R', 'ORBinf.L', 'ORBinf.R', 'SMA.L', 'IPL.L', 'IPL.R',
    'ANG.L', 'ANG.R', 'TPOsup.L', 'ITG.L'], # 18 III


    'Visual' : ['CAL.L', 'CAL.R', 'CUN.L', 'CUN.R', 'LING.L', 'LING.R', 'SOG.L', 'SOG.R', 'MOG.L', 'MOG.R', 
    'IOG.L', 'IOG.R', 'FFG.L', 'FFG.R'], # 14 II

    
    'Subcortical' : ['OLF.L', 'OLF.R', 'DCG.L', 'DCG.R', 'HIP.L', 'HIP.R', 'PHG.L', 'PHG.R',
    'AMYG.L', 'AMYG.R', 'CAU.L', 'CAU.R', 'PUT.L', 'PUT.R', 'PAL.L', 'PAL.R', 'THA.L', 'THA.R', 
    'TPOmid.L', 'TPOmid.R'], # 20 V
    }



subnetwork_sensorimotor = ['PreCG.L', 'PreCG.R', 'ROL.L', 'ROL.R',  'SMA.R', 'INS.L', 'INS.R', 
    'PoCG.L', 'PoCG.R', 'SPG.L', 'SPG.R',  'SMG.L', 'SMG.R', 
    'PCL.L', 'PCL.R', 'HES.L', 'HES.R', 'STG.L', 'STG.R', 'TPOsup.R']

subnetwork_defaultmode = ['SFGdor.L', 'SFGdor.R', 'ORBsup.R', 'SFGmed.L', 'SFGmed.R', 
    'ORBsupmed.L', 'ORBsupmed.R', 'REC.L', 'REC.R', 
    'ACG.L', 'ACG.R', 'PCG.L', 'PCG.R', 'PCUN.L', 'PCUN.R', 
    'MTG.L', 'MTG.R', 'ITG.R']

subnetwork_attention = ['ORBsup.L', 'MFG.L', 'MFG.R', 'ORBmid.L', 'ORBmid.R', 'IFGoperc.L', 'IFGoperc.R', 
    'IFGtriang.L', 'IFGtriang.R', 'ORBinf.L', 'ORBinf.R', 'SMA.L', 'IPL.L', 'IPL.R',
    'ANG.L', 'ANG.R', 'TPOsup.L', 'ITG.L']

subnetwork_visual = ['CAL.L', 'CAL.R', 'CUN.L', 'CUN.R', 'LING.L', 'LING.R', 'SOG.L', 'SOG.R', 'MOG.L', 'MOG.R', 
    'IOG.L', 'IOG.R', 'FFG.L', 'FFG.R']

subnetwork_subcortical = ['OLF.L', 'OLF.R', 'DCG.L', 'DCG.R', 'HIP.L', 'HIP.R', 'PHG.L', 'PHG.R',
    'AMYG.L', 'AMYG.R', 'CAU.L', 'CAU.R', 'PUT.L', 'PUT.R', 'PAL.L', 'PAL.R', 'THA.L', 'THA.R', 
    'TPOmid.L', 'TPOmid.R']




def segNetwork(networkList, ROI_name, ROI_marker):
    r'''
    networkList: 子网络节点名称列表list[list]
    ROI_name: AAL90顺序排列的节点名称
    ROI_marker: AAL90顺序排列的节点的影像标记物

    output: 与networkList对应的子网络影像标记物列表
    '''
    segWeightList = []
    for i in range(len(networkList)): # 遍历子网络
        subnetwork_weight = []

        for j in range(len(networkList[i])): # 取出每个子网络的影像标记物
            roiIndex = ROI_name.index(networkList[i][j])
            roiWeight = ROI_marker[roiIndex]
            subnetwork_weight.append(roiWeight)
        
        segWeightList.append(subnetwork_weight)
    return segWeightList



def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def intra_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def inter_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0



def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices