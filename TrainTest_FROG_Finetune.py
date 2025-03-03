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

from DataSetSplit import splitTrainTest, splitTrainTestKF, getCommonAdj, DataSetBase, DataSetRec, DataSetTrainAug
from Model_GNN import Finetune


def TrainTest_FROG(model, train_data, test_data, category_weight, paramDefault, saveModelPath):

    best_LR = paramDefault['LR']
    best_L2 = paramDefault['L2']
    best_BATCHSIZE = paramDefault['BATCHSIZE']
    best_EPOCHS = paramDefault['EPOCHS']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size = best_BATCHSIZE, num_workers = 0, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = best_BATCHSIZE, num_workers = 0, shuffle = False)

    criterion_cls = nn.CrossEntropyLoss(weight=torch.tensor(category_weight).type('torch.FloatTensor').to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=best_LR, weight_decay=best_L2)
    
    epochTrainLoss = []
    epochTestLoss = []
    epochMetric = []
    epochROC = []

    epochProb = []
    epochPred = []
    epochLabel = []

    # epochBiomarker = []
    epochId = []

    best_loss = 1000

    for epoch in range(best_EPOCHS):
        # train
        train_loss_list=[]
        model.train()
        for i, data in enumerate(train_loader):
            _, bold, adj_coef, label = data

            bold = bold.type('torch.FloatTensor')
            adj_coef = adj_coef.type('torch.FloatTensor')
            label = label.type('torch.LongTensor')
            bold, adj_coef, label = bold.to(device), adj_coef.to(device), label.to(device)
            
            logit, readout = model(bold, adj_coef)
            optimizer.zero_grad()

            train_loss = criterion_cls(logit, label)

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
        id_list = []

        with torch.no_grad():
            for data in test_loader:
                id_path, bold, adj_coef, label = data

                bold = bold.type('torch.FloatTensor')
                adj_coef = adj_coef.type('torch.FloatTensor')
                label = label.type('torch.LongTensor')
                bold, adj_coef, label = bold.to(device), adj_coef.to(device), label.to(device)
                logit, readout = model(bold, adj_coef)

                test_cls_loss = criterion_cls(logit, label).item()

                test_loss_list.append(test_cls_loss)

                # 算准确率
                pred = logit.argmax(1).cpu().numpy().tolist()
                pred_list += pred

                label = label.cpu().numpy().tolist()
                label_list += label

                # biomarker_list.append(g.cpu().numpy())
                id_list += list(id_path)

                onehot_list = []
                for index in range(len(label)):
                    onehot = np.zeros(2)
                    onehot[label[index]] = 1
                    onehot_list.append(onehot)
                label_onehot_list += onehot_list

                prob = F.softmax(logit, dim=-1).cpu().numpy().tolist()
                prob_list += prob
        
        ACC = accuracy_score(label_list, pred_list)
        SEN = recall_score(label_list, pred_list, pos_label = 1)
        SPE = recall_score(label_list, pred_list, pos_label = 0)

        fpr, tpr, _ = roc_curve(np.array(label_onehot_list)[:, 1], np.array(prob_list)[:, 1], pos_label=1)
        AUC = auc(fpr, tpr)

        epoch_train_loss = sum(train_loss_list) / len(train_loss_list)
        epoch_test_loss = sum(test_loss_list) / len(test_loss_list)
        
        epochTrainLoss.append(epoch_train_loss)
        epochTestLoss.append(epoch_test_loss)
        epochMetric.append([ACC, SEN, SPE, AUC])
        epochROC.append([fpr, tpr])

        epochProb.append(prob_list)
        epochPred.append(pred_list)
        epochLabel.append(label_list)

        epochId.append(id_list)

        print('Epoch {:03d} train loss {:f}  test loss {:f}  ACC {:f}  SEN {:f}  SPE {:f}  AUC {:f}'.format(
            epoch+1, epoch_train_loss, epoch_test_loss, ACC, SEN, SPE, AUC))

        
        gb = 1024**3
        total, used, free = shutil.disk_usage('./')
        total, used, free = total/gb, used/gb, free/gb
        if free > 2:

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
        

    return epochTrainLoss, epochTestLoss, epochMetric, epochId, epochLabel, epochPred




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configFile', default='./setting/FROG_Finetune_CN_MCI_ALL.yaml', type=str,
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
    modelPath = config['result']['modelPath']

    KF_datalist = splitTrainTestKF(K_num=5, dataPath=dataPath, sitenameList=sitenameList, categoryDict=categoryDict, random_state=0)
    for fidx in range(len(KF_datalist)):
        [kf_train_id, kf_train_label], [kf_test_id, kf_test_label] = KF_datalist[fidx]

        neg_num = Counter(kf_train_label)[0]
        pos_num = Counter(kf_train_label)[1]
        max_num = max(neg_num, pos_num)
        category_weight = [max_num/neg_num, max_num/pos_num]

        train_data = DataSetBase(id=kf_train_id, label=kf_train_label, adj_type=adj_type, adj_thr=adj_thr, adj_norm=True, bold_pad=True)
        test_data = DataSetBase(id=kf_test_id, label=kf_test_label, adj_type=adj_type, adj_thr=adj_thr, adj_norm=True, bold_pad=True)
        
        model = Finetune(
                 num_nodes=90, in_dim=1, embed_dim=8,
                 hidden_channels=32, kernel_list=[3,5,7,9], skip_channels=256, dropout=0.3, block_num=8,
                 readout_channels_1=512, readout_channels_2=32,
                 class_num=2
                 )


        load_params = torch.load(modelPath + '/onefold' + '/model_best_loss.pth')
        model_params = model.state_dict()
        same_params = {k: v for k, v in load_params.items() if k in model_params.keys()}
        model_params.update(same_params)
        model.load_state_dict(model_params)


        for name, param in model.named_parameters():
            # print('name: ', name)
            if 'backbone' in name:
                param.requires_grad = False

        # for name, param in model.named_parameters():
        #     print('name: ', name)
        #     print('param: ', param.requires_grad)
            # if 'backbone' in name:
            #     param.requires_grad = False

        currentPath = savePath + '/KF_' + str(fidx+1)

        print('\nKF_' + str(fidx+1))

        if not os.path.exists(currentPath):
            os.makedirs(currentPath) # 创建多级目录


        epochTrainLoss, epochTestLoss, \
        epochMetric, \
        epochId, epochLabel, epochPred = TrainTest_FROG(
                model=model, 
                train_data=train_data,
                test_data=test_data,
                category_weight = category_weight,
                paramDefault=paramDefault,
                saveModelPath=currentPath
                )
                
        
        np.save('{:s}/epochTrainLoss.npy'.format(currentPath), epochTrainLoss)
        np.save('{:s}/epochTestLoss.npy'.format(currentPath), epochTestLoss)
        np.save('{:s}/epochMetric.npy'.format(currentPath), epochMetric)
        np.save('{:s}/epochPred'.format(currentPath), epochPred)
        np.save('{:s}/epochLabel.npy'.format(currentPath), epochLabel)
        np.save('{:s}/epochId.npy'.format(currentPath), epochId)
