import os,sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d,Conv2d
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import gather_operation,grouping_operation
import torch.nn.functional as F
from torch.autograd import Variable
from models_PUGAN.ops_torch import mlp_conv, attention_unit, mlp

class Discriminator(nn.Module):
    def __init__(self,in_channels):
        super(Discriminator,self).__init__()
        
        self.start_number=32
        self.mlp_conv1=mlp_conv(in_channels=in_channels,layer_dim=[self.start_number, self.start_number * 2])
        self.attention_unit=attention_unit(in_channels=self.start_number*4)
        self.mlp_conv2=mlp_conv(in_channels=self.start_number*4,layer_dim=[self.start_number*4,self.start_number*8])
        self.mlp=mlp(in_channels=self.start_number*8,layer_dim=[self.start_number * 8, 1])

    def forward(self,inputs):
        features=self.mlp_conv1(inputs)
        features_global=torch.max(features,dim=2)[0] ##global feature
        features=torch.cat([features,features_global.unsqueeze(2).repeat(1,1,features.shape[2])],dim=1)
        features=self.attention_unit(features)

        features=self.mlp_conv2(features)
        features=torch.max(features,dim=2)[0]

        output=self.mlp(features)

        return output
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad