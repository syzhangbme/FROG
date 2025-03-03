import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml
import os
from collections import Counter
import shutil
from Utils import sce_loss

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
# import warnings
# warnings.filterwarnings('ignore')

SEED = 0
# random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from DataSetSplit import splitTrainTest, splitTrainTestKF, getCommonAdj, DataSetBase, DataSetRec
from Model_GNN import Autoencoder


def TrainTest_FROG(model, train_data, test_data, paramDefault, saveModelPath):

    best_LR = paramDefault['LR']
    best_L2 = paramDefault['L2']
    best_BATCHSIZE = paramDefault['BATCHSIZE']
    best_EPOCHS = paramDefault['EPOCHS']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size = best_BATCHSIZE, num_workers = 0, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = best_BATCHSIZE, num_workers = 0, shuffle = False)

    criterion_rec = sce_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=best_LR, weight_decay=best_L2)
    
    epochTrainLoss = []
    epochTestLoss = []
    
    best_loss = 1000

    for epoch in range(best_EPOCHS):
        # train
        train_loss_list=[]
        model.train()
        for i, data in enumerate(train_loader):
            _, trans, mask, rec, adj_coef, label = data

            trans = trans.type('torch.FloatTensor')
            mask = mask.type('torch.FloatTensor')
            rec = rec.type('torch.FloatTensor')
            adj_coef = adj_coef.type('torch.FloatTensor')
            label = label.type('torch.LongTensor')
            trans, mask, rec, adj_coef, label = trans.to(device), mask.to(device), rec.to(device), adj_coef.to(device), label.to(device)
            
            regress, embedding, _ = model(trans, adj_coef)
            optimizer.zero_grad()

            rec_loss = criterion_rec(regress*mask, rec*mask)
            train_loss = rec_loss

            train_loss_list.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            # print('train loss: ', train_loss.item())

        # test
        test_loss_list = []
        model.eval()
        pred_list = []
        prob_list = []
        label_list = []
        label_onehot_list = []

        mask_list = []
        rec_list = []
        biomarker_list = []
        id_list = []

        with torch.no_grad():
            for data in test_loader:

                id_path, trans, mask, rec, adj_coef, label = data

                trans = trans.type('torch.FloatTensor')
                mask = mask.type('torch.FloatTensor')
                rec = rec.type('torch.FloatTensor')
                adj_coef = adj_coef.type('torch.FloatTensor')
                label = label.type('torch.LongTensor')
                trans, mask, rec, adj_coef, label = trans.to(device), mask.to(device), rec.to(device), adj_coef.to(device), label.to(device)
                
                regress, embedding, _ = model(trans, adj_coef)

                test_rec_loss = criterion_rec(regress*mask, rec*mask).item()
                test_loss_list.append(test_rec_loss)

                # label = label.cpu().numpy().tolist()
                # label_list += label

                # mask_list.append(mask.cpu().numpy())
                # rec_list.append(rec.cpu().numpy())

                # biomarker_list.append(g.cpu().numpy())
                # id_list += list(id_path)

        epoch_train_loss = sum(train_loss_list) / len(train_loss_list)
        epoch_test_loss = sum(test_loss_list) / len(test_loss_list)
        
        epochTrainLoss.append(epoch_train_loss)
        epochTestLoss.append(epoch_test_loss)

        print('Epoch {:03d} train loss {:f} test loss {:f}'.format(epoch+1, epoch_train_loss, epoch_test_loss))

        
        gb = 1024**3
        total, used, free = shutil.disk_usage('./')
        total, used, free = total/gb, used/gb, free/gb
        if free > 2:
            # 每个epoch都保存模型
            # torch.save(model.state_dict(), '{:s}/model_{:d}.pth'.format(saveModelPath, epoch+1))

            # 保存最低loss的模型
            if epoch_test_loss < best_loss:
                if os.path.exists(saveModelPath + '/model_best_loss.pth'):
                    os.remove(saveModelPath + '/model_best_loss.pth')
                else:
                    pass

                best_loss = epoch_test_loss
                torch.save(model.state_dict(), saveModelPath + '/model_best_loss.pth')

        else:
            assert False, "磁盘空间不足！"
        

    return epochTrainLoss, epochTestLoss




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configFile', default='./setting/FROG_Autoencoder_CoRR.yaml', type=str,
                        help='configuration file for training and test.')
    args = parser.parse_args()

    config = None
    with open(args.configFile, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    dataPath = config['data']['dataPath'] # '../data/processed/augment/'
    sitenameList = config['data']['sitenameList']
    categoryDict = config['data']['categoryDict']
    adj_type = config['data']['adjType']
    adj_thr = config['data']['adjThr']

    paramDefault = config['model']['paramDefault']
    savePath = config['result']['savePath']


    onefold_datalist = splitTrainTest(test_size=0.2, dataPath=dataPath, sitenameList=sitenameList, categoryDict=categoryDict, random_state=0)
    [kf_train_id, kf_train_label], [kf_test_id, kf_test_label] = onefold_datalist[0]

    train_data = DataSetRec(id=kf_train_id, label=kf_train_label, adj_type=adj_type, adj_thr=adj_thr, adj_norm=True, bold_pad=True)
    test_data = DataSetRec(id=kf_test_id, label=kf_test_label, adj_type=adj_type, adj_thr=adj_thr, adj_norm=True, bold_pad=True)

    model = Autoencoder(
                num_nodes=90, in_dim=1, embed_dim=8,
                hidden_channels=32, kernel_list=[3,5,7,9], dropout=0.3, block_num=8,
                )
    
    currentPath = savePath + '/onefold'

    print('\nonefold')
    
    if not os.path.exists(currentPath):
        os.makedirs(currentPath) # 创建多级目录

    epochTrainLoss, epochTestLoss = TrainTest_FROG(
            model=model,
            train_data=train_data,
            test_data=test_data,
            paramDefault=paramDefault,
            saveModelPath=currentPath
            )
            
    np.save('{:s}/epochTrainLoss.npy'.format(currentPath), epochTrainLoss)
    np.save('{:s}/epochTestLoss.npy'.format(currentPath), epochTestLoss)
