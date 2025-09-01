import torch
import pandas as pd
import numpy as np

def collate_fn_ind(batch):
    # None 값을 필터링
    batch = [item for item in batch if item is not None]

    taxonomy_ids, model_ids, data = zip(*batch)

    data = torch.stack([torch.tensor(df.values, dtype=torch.float32) for df in data])


    return taxonomy_ids, model_ids, data

def collate_fn_sk(batch):
    # None 값을 필터링
    batch = [item for item in batch if item is not None]

    taxonomy_ids, model_ids, data, sk = zip(*batch)

    for df in data:
        data = torch.stack([torch.tensor(df.values, dtype=torch.float32)])
    
    for s in sk:
        sk = torch.stack([torch.tensor(s.values, dtype=torch.float32)])

    return taxonomy_ids, model_ids, data, sk

def collate_fn_seg(batch, npoints=8192):
    batch = [item for item in batch if item is not None]
    taxonomy_ids, model_ids, points = zip(*batch)

    padded_points = []

    for point in points:
        if len(point) < npoints:
            pad_size = npoints - len(point)
            point_data = np.pad(point, ((0, pad_size), (0, 0)), mode='constant')
        else:
            point_data = point[:npoints]
        padded_points.append(torch.tensor(point_data, dtype=torch.float32))

    point_tensor = torch.stack(padded_points)

    return taxonomy_ids, model_ids, point_tensor

def collate_fn_anchseg(batch, npoints=16384):
    batch = [item for item in batch if item is not None]
    model_ids, points = zip(*batch)

    padded_points = []

    for point in points:
        if len(point) < npoints:
            pad_size = npoints - len(point)
            point_data = np.pad(point, ((0, pad_size), (0, 0)), mode='constant')
        else:
            point_data = point[:npoints]
        padded_points.append(torch.tensor(point_data, dtype=torch.float32))

    point_tensor = torch.stack(padded_points)

    return model_ids, point_tensor

def collate_fn_seg_branch(batch, npoints=2048):
    batch = [item for item in batch if item is not None]
    taxonomy_ids, model_ids, points = zip(*batch)

    padded_points = []

    for point in points:
        if len(point) < npoints:
            pad_size = npoints - len(point)
            point_data = np.pad(point, ((0, pad_size), (0, 0)), mode='constant')
        else:
            point_data = point[:npoints]
        padded_points.append(torch.tensor(point_data, dtype=torch.float32))

    point_tensor = torch.stack(padded_points)

    return taxonomy_ids, model_ids, point_tensor

def collate_fn_pugan(batch, npoints=8192):
    batch = [item for item in batch if item is not None]
    model_ids, points = zip(*batch)

    padded_points = []
    
    for point in points:
        if len(point) < npoints:
            pad_size = npoints - len(point)
            point_data = np.pad(point, ((0, pad_size), (0, 0)), mode='constant')
        else:
            point_data = point[:npoints]
        padded_points.append(torch.tensor(point_data, dtype=torch.float32))

    point_tensor = torch.stack(padded_points)
    return model_ids, point_tensor