import torch
import torch.nn as nn
from models_PUGAN.generator import Generator
from models_PUGAN.discriminator import Discriminator
import json
from tools import builder
from utils import dist_utils, misc, misc_modified
from Common.loss_utils import get_uniform_loss, get_repulsion_loss, get_discriminator_loss, get_generator_loss, get_discriminator_loss_single, get_penalty
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from tools.to_tensor import collate_fn_pugan
from utils.misc_modified import fps
import os
from tqdm import tqdm
from glob import glob
import time
from termcolor import colored
import numpy as np

def xavier_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def rotate_point_cloud_torch(data):
    """
    Rotate each point cloud in the batch randomly along all 3 axes.
    Input: B x N x 3 (torch.Tensor)
    Output: B x N x 3 (rotated)
    """
    B = data.shape[0]
    device = data.device
    rotated_data = torch.zeros_like(data)

    for i in range(B):
        angles = torch.rand(3) * 2 * np.pi
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                           [0, torch.sin(angles[0]), torch.cos(angles[0])]], device=device)

        Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                           [0, 1, 0],
                           [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], device=device)

        Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                           [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                           [0, 0, 1]], device=device)

        R = torch.mm(Rz, torch.mm(Ry, Rx))  # (3,3)
        rotated_data[i] = torch.matmul(data[i], R)

    return rotated_data



def random_scale_point_cloud_torch(data, scale_low=0.9, scale_high=1.1):
    """
    Randomly scale each point cloud in the batch.
    Input: B x N x 3 (torch.Tensor)
    Output: B x N x 3 (scaled)
    """
    B = data.shape[0]
    scales = (scale_high - scale_low) * torch.rand(B, 1, 1, device=data.device) + scale_low

    return data * scales

def shift_point_cloud_torch(data, shift_range=0.1):
    """
    Randomly shift each point cloud in the batch.
    Input: B x N x 3 (torch.Tensor)
    Output: B x N x 3 (shifted)
    """
    B = data.shape[0]
    shifts = (2 * shift_range) * torch.rand(B, 1, 3, device=data.device) - shift_range
    return data + shifts


def jitter_perturbation_point_cloud(input_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, jittered point cloud
    """
    assert (clip > 0)
    jitter = np.clip(sigma * np.random.randn(*input_data.shape), -1 * clip, clip)
    jitter[:, 3:] = 0
    input_data += jitter
    return input_data

def run_PUGAN(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    train_dataloader.collate_fn = collate_fn_pugan
    test_dataloader.collate_fn = collate_fn_pugan
    # build model
    G_model = Generator()
    G_model.apply(xavier_init)

    D_model = Discriminator(in_channels = 3)
    D_model.apply(xavier_init)

    # base_model = builder.model_builder(config.model)
    if args.use_gpu:
        G_model.to(args.local_rank)
        D_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch_G = 0
    start_epoch_D = 0
    best_metrics = None
    best_metrics_G = None
    best_metrics_D = None
    metrics_G = None
    metrics_D = None
    

    # resume ckpts
    if args.resume:
        start_epoch_G, best_metrics_G = builder.resume_model_G(G_model, args, logger = logger)
        best_metrics_G = Metrics(config.consider_metric, best_metrics_G)
        start_epoch_D, best_metrics_D = builder.resume_model_D(D_model, args, logger = logger)
        best_metrics_D = Metrics(config.consider_metric, best_metrics_D)

        best_metrics = best_metrics_G
    elif args.start_ckpts is not None:
        builder.load_model(G_model, args.start_ckpts, logger = logger)
        builder.load_model(D_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in G_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    for name, param in D_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in G_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    for name, param in D_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            G_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G_model)
            D_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        G_model = nn.parallel.DistributedDataParallel(G_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        D_model = nn.parallel.DistributedDataParallel(D_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)        
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        G_model = nn.DataParallel(G_model).cuda()
        D_model = nn.DataParallel(D_model).cuda()
    # optimizer & scheduler
    optimizer_G = builder.build_optimizer(G_model, config)
    optimizer_D = builder.build_optimizer(D_model, config)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer_G(optimizer_G, args, logger = logger)
        builder.resume_optimizer_D(optimizer_D, args, logger = logger)
    scheduler_G = builder.build_scheduler(G_model, optimizer_G, config, last_epoch=start_epoch_G-1)
    scheduler_D = builder.build_scheduler(D_model, optimizer_D, config, last_epoch=start_epoch_D-1)
    # trainval
    # training
    G_model.zero_grad()
    D_model.zero_grad()
    
    for epoch in range(start_epoch_G, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        G_model.train()
        D_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['UniformLoss', 'CD2Loss', 'RepulsionLoss', 'GeneratorLoss', 'Penalty'])

        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME

            gt_data = data.cuda()

            # points, downsampling ratio, sigma
            input_data = misc_modified.point_cloud_down(gt_data, 0.25, 0.5, fixed_points=None, padding_zeros=False)
            input_data = input_data.cuda()
            
            num_iter += 1
           
            output_point_cloud = G_model(input_data.transpose(1,2))
            
            repulsion_loss = get_repulsion_loss(output_point_cloud.permute(0, 2, 1))
            uniform_loss = get_uniform_loss(output_point_cloud.permute(0, 2, 1))
            penalty_loss = get_penalty(output_point_cloud.permute(0, 2, 1))

            cd_loss_l2 = ChamferDisL2(output_point_cloud.transpose(1,2), gt_data)

            if config.use_gan == True:
                fake_pred = D_model(output_point_cloud.detach())
                d_loss_fake = get_discriminator_loss_single(fake_pred, label=False)
                d_loss_fake.backward()
                
                real_pred = D_model(gt_data.transpose(1,2).detach())
                d_loss_real = get_discriminator_loss_single(real_pred, label=True)
                d_loss_real.backward()

                d_loss = d_loss_real + d_loss_fake
                
                fake_pred = D_model(output_point_cloud)
                g_loss = get_generator_loss(fake_pred)

                total_G_loss = config.uniform_w * uniform_loss + config.emd_w * cd_loss_l2 + config.repulsion_w * repulsion_loss + config.gan_w * g_loss #+ config.emd_w * penalty_loss
            
            else:
                total_G_loss = config.emd_w * cd_loss_l2 + config.repulsion_w * repulsion_loss #+ penalty_loss

            total_G_loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(G_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                torch.nn.utils.clip_grad_norm_(D_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer_G.step()
                optimizer_D.step()
                
                G_model.zero_grad()
                D_model.zero_grad()

            if args.distributed:
                uniform_loss = dist_utils.reduce_tensor(uniform_loss, args)
                cd_loss_l2 = dist_utils.reduce_tensor(cd_loss_l2, args)
                repulsion_loss = dist_utils.reduce_tensor(repulsion_loss, args)
                g_loss = dist_utils.reduce_tensor(g_loss, args)
                penalty_loss = dist_utils.reduce_tensor(penalty_loss, args)

                losses.update([uniform_loss.item() * 1000, cd_loss_l2.item() * 1000, repulsion_loss.item() * 1000, g_loss.item() * 1000, penalty_loss.item() * 1000])
            else:
                losses.update([uniform_loss.item() * 1000, cd_loss_l2.item() * 1000, repulsion_loss.item() * 1000, g_loss.item() * 1000, penalty_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', uniform_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', cd_loss_l2.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer_G.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler_G.step()
                    scheduler_D.step()

        if isinstance(scheduler_D, list) or isinstance(scheduler_G, list):
            for item in scheduler_D:
                item.step()
            for item in scheduler_G:
                item.step()
        else:
            scheduler_D.step()
            scheduler_G.step()

        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(G_model, D_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(G_model, optimizer_G, epoch, metrics, best_metrics, 'ckpt-best_G', args, logger = logger)
                builder.save_checkpoint(D_model, optimizer_D, epoch, metrics, best_metrics, 'ckpt-best_D', args, logger = logger)
        builder.save_checkpoint(G_model, optimizer_G, epoch, metrics, best_metrics, 'ckpt-last_G', args, logger = logger)      
        builder.save_checkpoint(D_model, optimizer_D, epoch, metrics, best_metrics, 'ckpt-last_D', args, logger = logger)
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(G_model, optimizer_G, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}_G', args, logger = logger)
            builder.save_checkpoint(D_model, optimizer_D, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}_D', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(G_model, D_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    G_model.eval()  # set model to eval mode
    D_model.eval()

    test_losses = AverageMeter(['CDLossL1', 'CDLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, (model_ids, data) in enumerate(test_dataloader):
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            
            gt_data = data.cuda()
            
            input_data = misc_modified.point_cloud_down(gt_data, 0.25, 0.5, fixed_points=None, padding_zeros=False)
            input_data = input_data.cuda()
           
            output_point_cloud = G_model(input_data.transpose(1,2)).transpose(1,2) 

            cd_loss_l1 = ChamferDisL1(output_point_cloud, gt_data)
            cd_loss_l2 = ChamferDisL2(output_point_cloud, gt_data)
            
            if args.distributed:
                cd_loss_l1 = dist_utils.reduce_tensor(cd_loss_l1, args)
                cd_loss_l2 = dist_utils.reduce_tensor(cd_loss_l2, args)
                
            test_losses.update([cd_loss_l1.item() * 1000, cd_loss_l2.item() * 1000])
            
            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(output_point_cloud, gt_data)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]


            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        
        test_metrics.update(_metrics)

        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[-1]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                # test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc_modified.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc_modified.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)



                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc_modified.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 

