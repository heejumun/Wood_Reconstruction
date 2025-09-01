import torch
import torch.nn as nn
import os, sys
from pointnet2_ops import pointnet2_utils
import math
from knn_cuda import KNN
from utils.misc_modified import fps
from extensions.expansion_penalty.expansion_penalty_module import expansionPenaltyModule

# get_uniform_loss, get_repulsion_loss, discriminator_loss, generator_loss
knn_uniform = KNN(k=2, transpose_mode=True)
knn_repulsion=KNN(k=20, transpose_mode=True)

def get_uniform_loss(pcd, percentage=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
    B,N,C = pcd.shape[0], pcd.shape[1], pcd.shape[2]
    npoint = int(N*0.05)
    loss = 0
    new_xyz = fps(pcd.contiguous(), npoint) # B, C, N

    for p in percentage:
        nsample = int(N*p)
        r=math.sqrt(p*radius**2)
        disk_area = math.pi*(radius**2)*p/nsample

        idx = pointnet2_utils.ball_query(r, nsample, pcd.contiguous(), new_xyz.contiguous()) # b N nsample
        expect_len = math.sqrt(disk_area)
      
        grouped_pcd = pointnet2_utils.grouping_operation(pcd.permute(0,2,1).contiguous(),idx) # B C N nsample
        grouped_pcd = grouped_pcd.permute(0,2,3,1)

        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), dim=0) # B*N nsample C

        dist, _ = knn_uniform(grouped_pcd, grouped_pcd)
        uniform_dist = dist[:, :, 1:] #B*N nsample 1
        uniform_dist = torch.abs(uniform_dist + 1e-8)
        uniform_dist = torch.mean(uniform_dist, dim=1)
        uniform_dist = (uniform_dist - expect_len)**2/(expect_len+1e-8)

        mean_loss = torch.mean(uniform_dist)
        mean_loss = mean_loss * math.pow(p*100, 2)
        loss += mean_loss

    return loss/len(percentage)

def get_repulsion_loss(pcd, h=0.0005):
    dist, idx = knn_repulsion(pcd, pcd)

    dist = dist[:, :, 1:5] **2 # top 4 closest

    loss = torch.clamp(-dist+h, min=0)
    loss = torch.mean(loss)

    return loss

def get_discriminator_loss(pred_fake, pred_real):
    real_loss = torch.mean((pred_real-1)**2)
    fake_loss = torch.mean(pred_fake**2)
    loss = real_loss + fake_loss
    return loss

def get_generator_loss(pred_fake):
    fake_loss = torch.mean((pred_fake-1)**2)
    return fake_loss 

def get_discriminator_loss_single(pred, label = True):
    if label == True:
        loss = torch.mean((pred-1)**2)
        return loss
    else:
        loss = torch.mean((pred)**2)
        return loss
    
def get_penalty(pred_dense_points):
    penalty_func = expansionPenaltyModule()
    dist_dense, _, mean_mst_dis = penalty_func(pred_dense_points, 64, 1.2)
    loss_mst= torch.mean(dist_dense) *1000
    return loss_mst