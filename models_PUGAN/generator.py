import os,sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d,Conv2d
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import gather_operation,grouping_operation
import torch.nn.functional as F
from torch.autograd import Variable
from models_PUGAN.ops_torch import feature_extraction, up_projection_unit

class Generator(nn.Module):
    def __init__(self, params=None):
        super(Generator,self).__init__()
        self.feature_extractor=feature_extraction()
        #self.up_ratio=params['up_ratio']
        #self.num_points=params['patch_num_point']
        #self.out_num_point=int(self.num_points*self.up_ratio)
        self.up_projection_unit=up_projection_unit()

        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=3,kernel_size=1)
        )
    def forward(self,input):
        features=self.feature_extractor(input) #b,648,n


        H=self.up_projection_unit(features) #b,128,4*n

        coord=self.conv1(H)
        coord=self.conv2(coord)
        return coord

class Generator_recon(nn.Module):
    def __init__(self):
        super(Generator_recon,self).__init__()
        self.feature_extractor=feature_extraction()

        self.conv0=nn.Sequential(
            nn.Conv1d(in_channels=648,out_channels=128,kernel_size=1),
            nn.ReLU()
        )

        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=3,kernel_size=1)
        )
    def forward(self,input):
        features=self.feature_extractor(input) #b,648,n
        coord=self.conv0(features)
        coord=self.conv1(coord)
        coord=self.conv2(coord)
        return coord
