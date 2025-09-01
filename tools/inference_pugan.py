##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc_modified
from datasets.io import IO
from datasets.data_transforms import Compose
import torch
from pointnet2_ops import pointnet2_utils
import random

import open3d as o3d

def get_fscore(pred, gt, threshold=0.01):
    """
    pred, gt: torch.Tensor of shape (1, N, 3)
    threshold: float, distance threshold
    returns: torch scalar
    """
    b = pred.size(0)
    device = pred.device
    if b != 1:
        scores = [get_fscore(pred[i:i+1], gt[i:i+1], threshold) for i in range(b)]
        return torch.tensor(scores).mean().to(device)

    # convert to open3d point cloud
    pred_np = pred.squeeze(0).detach().cpu().numpy()
    gt_np = gt.squeeze(0).detach().cpu().numpy()
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_np)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_np)

    dist1 = pcd_pred.compute_point_cloud_distance(pcd_gt)
    dist2 = pcd_gt.compute_point_cloud_distance(pcd_pred)

    recall = sum(d < threshold for d in dist2) / len(dist2)
    precision = sum(d < threshold for d in dist1) / len(dist1)

    fscore = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return torch.tensor(fscore).to(device)

from extensions.chamfer_dist import ChamferFunction

def chamfer_l2(pred, gt):
    """
    pred, gt: torch.Tensor of shape (1, N, 3)
    returns: scalar chamfer distance
    """
    # nan 제거
    pred = pred[~torch.isnan(pred).any(dim=2)].reshape(1, -1, 3)
    gt = gt[~torch.isnan(gt).any(dim=2)].reshape(1, -1, 3)

    dist1, dist2 = ChamferFunction.apply(pred, gt)
    return torch.mean(dist1) + torch.mean(4 * dist2)

def point_cloud_down(xyz, ratio, center_sigma, fixed_points=None, padding_zeros=False):
    '''
    Separate point cloud: generate the incomplete point cloud with a set number of points.
    Input:
        xyz: torch.Tensor (B, num_points, 3) with 0-padding
        num_points: 원래 target point 수 (예: 8192)
        num_part: 마스킹할 part 수
    Output:
        crop_data: (B, num_points, 3)
        masked_data: (B, num_points, 3)
    '''

    random.seed(42)
    torch.manual_seed(42)
        
    N = xyz.shape[0]
    target_N = int(N * ratio)
    center = xyz[torch.randint(0, N, (1,)).item()]
    dists = torch.linalg.norm(xyz - center, axis=1)
    prob = torch.exp(dists / center_sigma)
    prob = prob / prob.sum()
    sampled_idx = torch.multinomial(prob, target_N, replacement=False)
    
    crop_data = xyz[sampled_idx]
    crop_data = crop_data.unsqueeze(0)

    return crop_data.contiguous()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def farthest_point_sampling(xyz, num_samples):
    xyz = xyz.contiguous().cuda()
    center = fps(xyz, int(num_samples))  # B G 3
    return center



def inference_single(G_model, pc_path, args, config,  fscore_list, cd_list, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    model_ids = pc_path.split('.')[0]
    # read single point cloud
    pc_ndarray = IO.get(pc_file)

    # transform it according to the model 
    # normalize it to fit the model on ShapeNet-55/34
    pc_ndarray = pc_ndarray.replace({'\ufeff': ''}, regex=True)
    pc_ndarray = pc_ndarray.astype(np.float32)
    
    mask_wood = (pc_ndarray.iloc[:, 3] == 2)
    pc_ndarray = pc_ndarray.loc[mask_wood, :].iloc[:, :3]  # 앞 3개 컬럼만

    centroid = np.nanmean(pc_ndarray, axis=0)
    pc_ndarray = pc_ndarray - centroid
    m = np.max(np.sqrt(np.nansum(pc_ndarray**2, axis=1)))
    pc_ndarray = pc_ndarray / m
    
    pc_ndarray = torch.tensor(pc_ndarray.values)

    gt_data = pc_ndarray
    
    if gt_data.shape[0] > 15000:
        indices = torch.randperm(gt_data.shape[0])[:15000]
        gt_data = gt_data[indices]

    input_data = gt_data.cuda()
    print(input_data.shape)
    input_data = point_cloud_down(input_data, 1, 0.5, fixed_points=None, padding_zeros=False)
    input_data = input_data.cuda()
    
    if torch.isnan(input_data).any() or torch.isinf(input_data).any():
        print("[ERROR] point_cloud contains NaN or Inf")
    # transform = Compose([{
    #     'callback': 'UpSamplePoints',
    #     'parameters': {
    #         'n_points': 163840
    #     },
    #     'objects': ['input']
    # }, {
    #     'callback': 'ToTensor',
    #     'objects': ['input']
    # }])

    final_image = []

    # reconstructed
    output_point_cloud = G_model(input_data.transpose(1,2)).transpose(1,2)
    dense_points = output_point_cloud.squeeze(0).detach().cpu().numpy()
    # ===== Metric 계산 =====
    gt_tensor = torch.tensor(gt_data, dtype=torch.float32).unsqueeze(0).to(args.device)
    pred_tensor = torch.tensor(dense_points, dtype=torch.float32).unsqueeze(0).to(args.device)

    if pred_tensor.shape[1] > 0 and gt_tensor.shape[1] > 0:
        fscore = get_fscore(pred_tensor, gt_tensor)
        cd = chamfer_l2(pred_tensor, gt_tensor)

        fscore_list.append(fscore.item())
        cd_list.append(cd.item())
    else:
        print(f"[WARNING] 빈 point cloud가 존재합니다: {pc_path}")

    input_data = input_data.squeeze(0).detach().cpu().numpy()
    gt_data = gt_data.squeeze(0).detach().cpu().numpy()

    # denormalize it to adapt for the original input
    dense_points = dense_points * m
    dense_points = dense_points + centroid

    input_data = input_data * m
    input_data = input_data + centroid

    gt_data = gt_data * m
    gt_data = gt_data + centroid

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)
        np.savetxt(os.path.join(target_path, 'gt.csv'), gt_data.squeeze(), delimiter=',')
        np.savetxt(os.path.join(target_path, 'input.csv'), input_data.squeeze(), delimiter=',')
        np.savetxt(os.path.join(target_path, 'fine.csv'), dense_points.squeeze(), delimiter=',')

        if args.save_vis_img:
            final_image = []

            # 1) Partial + Masked (Ground Truth)
            img_partial_masked = misc_modified.get_ptcloud_img_dual(
                partial=input_data,
                other=gt_data,
                other_color='red',
                roll=0,
                pitch=0
            )
            final_image.append(img_partial_masked)

            # 2) Partial + Predicted (Completion Result)
            img_partial_predicted = misc_modified.get_ptcloud_img_dual(
                partial=input_data,
                other=dense_points,
                other_color='green',
                roll=0,
                pitch=0
            )
            final_image.append(img_partial_predicted)

            # concatenate and save
            img = np.concatenate(final_image, axis=1)
            cv2.imwrite(os.path.join(target_path, 'input.jpg'), img)

    return

def main():
    fscore_list = []
    cd_list = []
    from models_PUGAN.generator import Generator
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    G_model = Generator()
    builder.load_model(G_model, args.model_checkpoint)
    G_model.to(args.device.lower())
    G_model.eval()

    if args.pc_root != '':
        pc_file_list = os.listdir(args.pc_root)
        for pc_file in pc_file_list:
            inference_single(G_model, pc_file, args, config, fscore_list=fscore_list, cd_list=cd_list, root=args.pc_root)
        print("\n===== Evaluation Summary =====")
        print(f"Mean F1-Score (th=0.01): {np.mean(fscore_list):.4f}")
        print(f"Mean Chamfer Distance: {np.mean(cd_list):.6f}")

    else:
        inference_single(G_model, args.pc, args, config)

if __name__ == '__main__':
    main()