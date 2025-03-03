import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout):
        super(gcn,self).__init__()
        # self.gconv = gconv()
        c_in = 3*c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = dropout

    def forward(self,x,A):
        out = [x]
        
        # x1 = self.gconv(x,A)
        x1 = torch.einsum('ncvt, nvw->ncwt',(x,A))
        x1 = x1.contiguous()
        out.append(x1)
        
        # x2 = self.gconv(x1,A)
        x2 = torch.einsum('ncvt, nvw->ncwt',(x1,A))
        x2 = x2.contiguous()
        out.append(x2)

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout)
        return h



class t_branch(nn.Module):
    def __init__(self,c_in, c_out_split, kernel_size):
        super(t_branch, self).__init__()

        self.filter_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out_split,
                                    kernel_size=(1,kernel_size), dilation=1, padding='same')
        
        self.gate_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out_split,
                                    kernel_size=(1, kernel_size), dilation=1, padding='same')
        
    def forward(self, x): # B C V T
        filter = self.filter_conv(x)
        filter = torch.tanh(filter)
        gate = self.gate_conv(x)
        gate = torch.sigmoid(gate)

        t = filter * gate
        return t


class tcn(nn.Module):
    def __init__(self, c_in, c_out, kernel_list):
        super(tcn, self).__init__()

        c_out_split = c_out//len(kernel_list)
        self.conv_1 = t_branch(c_in, c_out_split, kernel_list[0])
        self.conv_2 = t_branch(c_in, c_out_split, kernel_list[1])
        self.conv_3 = t_branch(c_in, c_out_split, kernel_list[2])
        self.conv_4 = t_branch(c_in, c_out_split, kernel_list[3])

    def forward(self, x):
        branch_1 = self.conv_1(x)
        branch_2 = self.conv_2(x)
        branch_3 = self.conv_3(x)
        branch_4 = self.conv_4(x)

        branch_merge = torch.cat([branch_1, branch_2, branch_3, branch_4], dim=1)

        return branch_merge


class dt_branch(nn.Module):
    def __init__(self, c_in, c_out_split, kernel_size):
        super(dt_branch, self).__init__()
        self.filter_conv = nn.ConvTranspose2d(in_channels=c_in,
                                    out_channels=c_out_split,
                                    kernel_size=(1,kernel_size), dilation=1)

        self.gate_conv = nn.ConvTranspose2d(in_channels=c_in,
                                    out_channels=c_out_split,
                                    kernel_size=(1, kernel_size), dilation=1)
        
    def forward(self, x): # B C V T
        filter = self.filter_conv(x)
        filter = torch.tanh(filter)
        gate = self.gate_conv(x)
        gate = torch.sigmoid(gate)

        t = filter * gate
        return t


class dtcn(nn.Module):
    def __init__(self, c_in, c_out, kernel_list):
        super(dtcn, self).__init__()
        self.kernel_list = kernel_list
        c_out_split = c_out//len(kernel_list)
        self.conv_1 = dt_branch(c_in, c_out_split, kernel_list[0])
        self.conv_2 = dt_branch(c_in, c_out_split, kernel_list[1])
        self.conv_3 = dt_branch(c_in, c_out_split, kernel_list[2])
        self.conv_4 = dt_branch(c_in, c_out_split, kernel_list[3])

    def forward(self, x):

        # print('x shape: ', x.shape)
        branch_1 = self.conv_1(x)[:, :, :, (self.kernel_list[0]-1)//2 : -(self.kernel_list[0]-1)//2]
        branch_2 = self.conv_2(x)[:, :, :, (self.kernel_list[1]-1)//2 : -(self.kernel_list[1]-1)//2]
        branch_3 = self.conv_3(x)[:, :, :, (self.kernel_list[2]-1)//2 : -(self.kernel_list[2]-1)//2]
        branch_4 = self.conv_4(x)[:, :, :, (self.kernel_list[3]-1)//2 : -(self.kernel_list[3]-1)//2]

        # print('branch_1 shape: ', branch_1.shape)
        # print('branch_2 shape: ', branch_2.shape)
        # print('branch_3 shape: ', branch_3.shape)
        # print('branch_4 shape: ', branch_4.shape)

        branch_merge = torch.cat([branch_1, branch_2, branch_3, branch_4], dim=1)

        return branch_merge




class En_STBlock(nn.Module):
    def __init__(self, hidden_channels, kernel_list, dropout):
        super(En_STBlock,self).__init__()

        self.tconv = tcn(c_in=hidden_channels, c_out=hidden_channels, kernel_list=kernel_list)
        self.gconv = gcn(c_in=hidden_channels, c_out=hidden_channels, dropout=dropout)
        self.bn = nn.BatchNorm2d(hidden_channels)

        # self.out_convs = nn.Conv2d(in_channels=hidden_channels, out_channels=skip_channels, kernel_size=(1, 1))


    def forward(self, X, adj): # B C V T

        residual = X

        X = self.tconv(X)
        X = self.gconv(X, adj)

        # X = X + residual[:, :, :, -X.size(3):] # 残差裁剪
        X = X + residual
        X = self.bn(X)

        return X

        # out = self.out_convs(X)

        # return X, out




class De_STBlock(nn.Module):
    def __init__(self, hidden_channels, kernel_list, dropout):
        super(De_STBlock,self).__init__()

        self.dtconv = dtcn(c_in=hidden_channels, c_out=hidden_channels, kernel_list=kernel_list)
        self.gconv = gcn(c_in=hidden_channels, c_out=hidden_channels, dropout=dropout)
        self.bn = nn.BatchNorm2d(hidden_channels)
        
        # self.out_convs = nn.Conv2d(in_channels=hidden_channels, out_channels=skip_channels, kernel_size=(1, 1))

        # self.respad = nn.ReflectionPad2d(padding=(kernel_size-1, 0, 0, 0)) # 左右上下


    def forward(self, X, adj): # B C V T

        # residual = self.respad(X) # 残差填充
        residual = X

        X = self.dtconv(X)
        X = self.gconv(X, adj)

        X = X + residual # 残差
        X = self.bn(X)

        return X

        # out = self.out_convs(X)

        # return X, out




class Autoencoder(nn.Module):
    def __init__(self,
                 num_nodes=90, in_dim=1, embed_dim=8,
                 hidden_channels=32, kernel_list=[3,5,7,9], dropout=0.3, block_num=8,
                 ):
        super(Autoencoder, self).__init__()

        # self.kernel_size = kernel_size

        self.en_stblock = nn.ModuleList()
        for b in range(block_num):
            self.en_stblock.append(En_STBlock(hidden_channels, kernel_list, dropout))

        self.de_stblock = nn.ModuleList()
        for b in range(block_num):
            self.de_stblock.append(De_STBlock(hidden_channels, kernel_list, dropout))

        self.down_dim = nn.Conv2d(in_channels=hidden_channels, out_channels=embed_dim, kernel_size=(1,1))
        self.up_dim = nn.Conv2d(in_channels=embed_dim, out_channels=hidden_channels, kernel_size=(1,1))

        self.start_stconv = nn.Conv2d(in_channels=in_dim, out_channels=hidden_channels, kernel_size=(1,1))
        self.end_stconv = nn.Conv2d(in_channels=hidden_channels, out_channels=in_dim, kernel_size=(1,1))


    def forward(self, input, adj): # B V T

        input = torch.unsqueeze(input, dim=1) # B C V T
        X = self.start_stconv(input)

        # T = X.shape[3]
        # T_min = T - len(self.en_stblock) * (self.kernel_size-1)

        out_vec_list = []
        
        for i in range(len(self.en_stblock)):
            X = self.en_stblock[i](X, adj)
            # out_vec_list.append(X[:, :, :, -T_min:]) # 不同尺度的ST块输出，裁剪成相同长度
            out_vec_list.append(X)
        
        embedding = self.down_dim(X)

        X = self.up_dim(embedding)

        for j in range(len(self.de_stblock)):
            X = self.de_stblock[j](X, adj)

        regress = self.end_stconv(X)
        regress = regress.squeeze()

        return regress, embedding, out_vec_list




class Finetune(nn.Module):
    def __init__(self,
                 num_nodes=90, in_dim=1, embed_dim=8,
                 hidden_channels=32, kernel_list=[3,5,7,9], skip_channels=256, dropout=0.3, block_num=8,
                 readout_channels_1=512, readout_channels_2=32,
                 class_num=2
                 ):
        super(Finetune, self).__init__()

        # self.kernel_size = kernel_list
        self.block_num = block_num

        self.backbone = Autoencoder(
                 num_nodes=num_nodes, in_dim=in_dim, embed_dim=embed_dim,
                 hidden_channels=hidden_channels, kernel_list=kernel_list, dropout=dropout, block_num=block_num,
                 )
        
        self.outconv = nn.ModuleList()
        for s in range(block_num):
            self.outconv.append(nn.Conv2d(in_channels=hidden_channels, out_channels=skip_channels, kernel_size=(1, 1)))

        self.readout_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=readout_channels_1, kernel_size=(1,1), bias=True)
        self.readout_conv_2 = nn.Conv2d(in_channels=readout_channels_1, out_channels=readout_channels_2, kernel_size=(1,1), bias=True)

        self.clf = nn.Sequential(
            nn.Linear(num_nodes*readout_channels_2, 32),
            nn.Linear(32, class_num)
        )
    

    def forward(self, input, adj): # B V T

        # input = torch.unsqueeze(input, dim=1) # B C V T
        regress, embedding, out_vec_list = self.backbone(input, adj)

        out_vec = 0

        for k in range(len(out_vec_list)):
            X_temp = out_vec_list[k]
            s = self.outconv[k](X_temp)
            out_vec = out_vec + s

        readout = F.relu(out_vec)
        readout = F.relu(self.readout_conv_1(readout))
        readout = self.readout_conv_2(readout)

        readout = torch.mean(readout, dim=3, keepdim=False) # B C V T -> BCV
        readout = readout.reshape(readout.size(0), -1) # B C*V
        logit = self.clf(readout)

        return logit, readout

