import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN


def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc

def random_sample(data, number):
    '''
        data B N 3
        number int
    '''
    assert data.size(1) > number
    assert len(data.shape) == 3
    ind = torch.multinomial(torch.rand(data.size()[:2]).float(), number).to(data.device)
    data = torch.gather(data, 1, ind.unsqueeze(-1).expand(-1, -1, data.size(-1)))
    return data

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_lambda_sche(opti, config, last_epoch=-1):
    if config.get('decay_step') is not None:
        # lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        warming_up_t = getattr(config, 'warmingup_e', 0)
        lr_lbmd = lambda e: max(config.lr_decay ** ((e - warming_up_t) / config.decay_step), config.lowest_decay) if e >= warming_up_t else max(e / warming_up_t, 0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, config, last_epoch=-1):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return bnm_scheduler
    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)

def farthest_point_sampling(xyz, num_samples):
    xyz = xyz.contiguous().cuda()
    center = fps(xyz, int(num_samples))  # B G 3
    return center

def mask_center_rand(center, num_parts, mask_ratio):
    '''
        center : B G 3
        mask : B G (bool)
    '''

    if mask_ratio == 0:
        return torch.zeros(center.shape[:2]).bool().to(center.device)

    mask_idx = []
    B, G, _ = center.shape  # B: 배치 크기, G: 점 개수
    
    for b in range(B):  # 배치별로 처리
        points = center[b].unsqueeze(0)  # 1 G 3
        mask = torch.zeros(G, dtype=torch.bool, device=center.device)  # G개의 마스크 초기화
        used_indices = set()  # 이미 선택된 인덱스 기록

        for _ in range(int(num_parts)):  
            # 랜덤으로 중심점 선택
            while True:
                index = random.randint(0, G - 1)
                if index not in used_indices:
                    break

            # 중심점과의 거리 계산
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)  # 1 G
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0].tolist()  # G, 가까운 순으로 정렬

            # 중복 제거
            idx = [i for i in idx if i not in used_indices]

            # 마스크할 점 개수 계산
            part_ratio = mask_ratio / num_parts
            mask_num = max(1, int(part_ratio * G))  # 최소 1개의 점은 항상 마스크

            # 부족한 점 채우기
            while len(idx) < mask_num:
                # 새로운 중심점 선택
                while True:
                    new_index = random.randint(0, G - 1)
                    if new_index not in used_indices:
                        break

                # 새로운 중심점과의 거리 계산
                new_distance_matrix = torch.norm(points[:, new_index].reshape(1, 1, 3) - points, p=2, dim=-1)  # 1 G
                new_idx = torch.argsort(new_distance_matrix, dim=-1, descending=False)[0].tolist()

                # 중복 제거 후 idx에 추가
                new_idx = [i for i in new_idx if i not in used_indices]
                idx.extend(new_idx)

            # 최종 마스크 업데이트
            mask[idx[:mask_num]] = True
            used_indices.update(idx[:mask_num])  # 사용된 인덱스 기록

        mask_idx.append(mask)

    # 배치 차원을 포함하여 스택
    bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G
    return bool_masked_pos



def process_point_cloud(xyz, num_parts, mask_ratio=0.4):
    B, N, _ = xyz.shape
    xyz = xyz.cuda()

    # 마스크 생성
    bool_masked_pos = mask_center_rand(xyz, num_parts, mask_ratio)

    # 마스킹되지 않은 점들 추출
    neighborhoods = xyz[~bool_masked_pos].view(B, -1, 3).cuda()
    masked = xyz[bool_masked_pos].view(B, -1, 3).cuda()
    return neighborhoods, masked


def seprate_point_cloud(xyz, num_points, num_part, fixed_points=None, padding_zeros=False):
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
    B, n, c = xyz.shape

    assert n == num_points, f"Expected {num_points}, but got {n}"
    assert c == 3, "Point cloud data must have 3 coordinates"

    CROP = []
    MASKED = []

    for points in xyz:
        # 0으로 padding된 점 제거
        mask_nonzero = ~(points == 0).all(dim=1)
        valid_points = points[mask_nonzero]  # (n_valid, 3)

        # 배치 차원 추가해서 process
        valid_points = valid_points.unsqueeze(0)  # (1, n_valid, 3)
        crop_data, masked = process_point_cloud(valid_points, num_part)  # 출력도 (1, m, 3)
        # 다시 (n, 3) 형태로
        crop_data = crop_data.squeeze(0)
        masked = masked.squeeze(0)

        # 부족한 부분 0으로 padding
        pad_len_crop = int(num_points - crop_data.shape[0])
        if pad_len_crop > 0:
            pad_crop = torch.zeros((pad_len_crop, 3), device=crop_data.device)
            crop_data = torch.cat([crop_data, pad_crop], dim=0)

        pad_len_mask = int(num_points - masked.shape[0])
        if pad_len_mask > 0:
            pad_masked = torch.zeros((pad_len_mask, 3), device=masked.device)
            masked = torch.cat([masked, pad_masked], dim=0)

        # 배치 차원 복원
        CROP.append(crop_data.unsqueeze(0))
        MASKED.append(masked.unsqueeze(0))

    crop_data = torch.cat(CROP, dim=0)  # (B, 8192, 3)
    masked_data = torch.cat(MASKED, dim=0)

    return crop_data.contiguous(), masked_data.contiguous()

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
    B, n, c = xyz.shape

    CROP = []
    MASKED = []

    for points in xyz:
        # 0으로 padding된 점 제거
        mask_nonzero = ~(points == 0).all(dim=1)
        valid_points = points[mask_nonzero]  # (n_valid, 3)
        
        N = valid_points.shape[0]
        target_N = int(N * ratio)
        center = valid_points[torch.randint(0, N, (1,)).item()]
        dists = torch.linalg.norm(valid_points - center, axis=1)
        prob = torch.exp(dists / center_sigma)
        prob = prob / prob.sum()
        sampled_idx = torch.multinomial(prob, target_N, replacement=False)
        
        crop_data = valid_points[sampled_idx]
        # 부족한 부분 0으로 padding
        pad_len_crop = int(n - crop_data.shape[0])
        if pad_len_crop > 0:
            pad_crop = torch.zeros((pad_len_crop, 3), device=crop_data.device)
            crop_data = torch.cat([crop_data, pad_crop], dim=0)

        # 배치 차원 복원
        CROP.append(crop_data.unsqueeze(0))

    crop_data = torch.cat(CROP, dim=0)  # (B, 8192, 3)

    return crop_data.contiguous()

def get_ptcloud_img(ptcloud,roll,pitch):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.add_subplot(111, projection='3d', adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(roll,pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=y, cmap='jet', s=0.2)

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img

def get_ordered_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(30, 45)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    
    num_point = ptcloud.shape[0]
    num_part = 14
    num_pt_per_part = num_point//num_part
    colors = np.zeros([num_point])
    delta_c = abs(1.0/num_part)
    print()
    for j in range(ptcloud.shape[0]):
        part_n = j//num_pt_per_part
        colors[j] = part_n*delta_c
        # print(colors[j,:])

    ax.scatter(x, y, z, zdir='z', c=colors, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    
    plt.close(fig)
    return img

def get_ptcloud_img_dual(partial, other, other_color, roll=0, pitch=0):
    """
    partial : (N,3) numpy array
    other : (M,3) numpy array (masked or predicted)
    other_color : str ('red' or 'green')
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d', adjustable='box')
    ax.axis('off')
    ax.view_init(roll, pitch)

    # 점 크기
    s=0.5

    # Partial - 파랑
    if partial is not None and len(partial)>0:
        x, z, y = partial.transpose(1,0)
        ax.scatter(x, y, z, c='blue', s=s, label='Partial')

    # Other - 지정색
    if other is not None and len(other)>0:
        x, z, y = other.transpose(1,0)
        ax.scatter(x, y, z, c=other_color, s=s, label='Other')

    # 범위 자동설정
    all_points = np.concatenate([arr for arr in [partial, other] if arr is not None])
    max_coord, min_coord = np.max(all_points), np.min(all_points)
    ax.set_xbound(min_coord, max_coord)
    ax.set_ybound(min_coord, max_coord)
    ax.set_zbound(min_coord, max_coord)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img





def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y', 
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    

def random_scale(partial, gt, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale, gt * scale



from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
