import numpy as np
import pandas as pd
import torch
import copy
from scipy import sparse
import pdb
import sys
sys.path.append('/home/jongwon/Desktop/CARLA_benchmark_project/LaneGCN/')

from utils import Logger, load_pretrain, gpu
from cargoverse_data import ArgoDataset, ArgoTestDataset, from_numpy, ref_copy

def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs

def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data

def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data

def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data



def preprocess(graph, cross_dist, cross_angle=None, data=None):
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    if cross_angle is not None:
        f1 = graph['feats'][hi]
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < config['cross_angle'])
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -config['cross_angle'])
        right_mask = mask.logical_not()

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    return out

def get_preprocessed_data(map_name, loc_dict, cam, curr_id):
    # 1. ArgoData.read_argo_data
    city = map_name[0]+map_name[-2:]
    df = pd.DataFrame.from_dict(loc_dict)

    agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
    mapping = dict()
    for i, ts in enumerate(agt_ts):
        mapping[ts] = i

    trajs = np.concatenate((
        df.X.to_numpy().reshape(-1, 1),
        df.Y.to_numpy().reshape(-1, 1)), 1)

    steps = [mapping[x] for x in df['TIMESTAMP'].values]
    steps = np.asarray(steps, np.int64)

    
    objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
    keys = list(objs.keys())
    obj_type = [x[1] for x in keys]
    vids = [x[0] for x in keys]

    ctx_trajs, ctx_steps = [], []
    for key in keys:
        idcs = objs[key]
        ctx_trajs.append(trajs[idcs])
        ctx_steps.append(steps[idcs])

    data = dict()
    data['city'] = city
    data['trajs'] = ctx_trajs
    data['steps'] = ctx_steps

    # 2. ArgoData.get_obj_feats

    orig = data['trajs'][0][19].copy().astype(np.float32)

    pre = data['trajs'][0][18] - orig
    theta = np.pi - np.arctan2(pre[1], pre[0])

    rot = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], np.float32)

    feats, ctrs, gt_preds, has_preds = [], [], [], []
    for traj, step in zip(data['trajs'], data['steps']):
        if 19 not in step:
            continue

        gt_pred = np.zeros((30, 2), np.float32)
        has_pred = np.zeros(30, np.bool)
        future_mask = np.logical_and(step >= 20, step < 50)
        post_step = step[future_mask] - 20
        post_traj = traj[future_mask]
        gt_pred[post_step] = post_traj
        has_pred[post_step] = 1

        obs_mask = step < 20
        step = step[obs_mask]
        traj = traj[obs_mask]
        idcs = step.argsort()
        step = step[idcs]
        traj = traj[idcs]

        for i in range(len(step)):
            if step[i] == 19 - (len(step) - 1) + i:
                break
        step = step[i:]
        traj = traj[i:]

        feat = np.zeros((20, 3), np.float32)
        feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
        feat[step, 2] = 1.0


        #LaneGCN Default Setting (Pred Range) -> can be change to CARLA
        x_min, x_max, y_min, y_max = [-200.0, 200.0, -200.0, 200.0] 

        if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
            continue

        ctrs.append(feat[-1, :2].copy())
        feat[1:, :2] -= feat[:-1, :2]
        feat[step[0], :2] = 0
        feats.append(feat)
        gt_preds.append(gt_pred)
        has_preds.append(has_pred)

    feats = np.asarray(feats, np.float32)
    ctrs = np.asarray(ctrs, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    has_preds = np.asarray(has_preds, np.bool)

    data['feats'] = feats
    data['ctrs'] = ctrs
    data['orig'] = orig
    data['theta'] = theta
    data['rot'] = rot
    data['gt_preds'] = gt_preds
    data['has_preds'] = has_preds
    data['idx'] = curr_id

    # 3. Argo_data.get_lane_graph
    """Get a rectangle area defined by pred_range."""
    radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
    lane_ids = cam.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
    lane_ids = copy.deepcopy(lane_ids)

    lanes = dict()
    for lane_id in lane_ids:
        lane = cam.city_lane_centerlines_dict[data['city']][lane_id]
        lane = copy.deepcopy(lane)
        centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
        x, y = centerline[:, 0], centerline[:, 1]
        if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
            continue
        else:
            """Getting polygons requires original centerline"""
            polygon = cam.get_lane_segment_polygon_2d(lane_id, data['city'])
            polygon = copy.deepcopy(polygon)
            lane.centerline = centerline
            lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
            lanes[lane_id] = lane

    lane_ids = list(lanes.keys())
    ctrs, feats, turn, control, intersect = [], [], [], [], []
    for lane_id in lane_ids:
        lane = lanes[lane_id]
        ctrln = lane.centerline
        num_segs = len(ctrln) - 1

        ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
        feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

        x = np.zeros((num_segs, 2), np.float32)
        if lane.turn_direction == 'LEFT':
            x[:, 0] = 1
        elif lane.turn_direction == 'RIGHT':
            x[:, 1] = 1
        else:
            pass
        turn.append(x)

        control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
        intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

    node_idcs = []
    count = 0
    for i, ctr in enumerate(ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)
    num_nodes = count

    pre, suc = dict(), dict()
    for key in ['u', 'v']:
        pre[key], suc[key] = [], []
    for i, lane_id in enumerate(lane_ids):
        lane = lanes[lane_id]
        idcs = node_idcs[i]

        pre['u'] += idcs[1:]
        pre['v'] += idcs[:-1]
        if lane.predecessors is not None:
            for nbr_id in lane.predecessors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre['u'].append(idcs[0])
                    pre['v'].append(node_idcs[j][-1])

        suc['u'] += idcs[:-1]
        suc['v'] += idcs[1:]
        if lane.successors is not None:
            for nbr_id in lane.successors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc['u'].append(idcs[-1])
                    suc['v'].append(node_idcs[j][0])

    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
    for i, lane_id in enumerate(lane_ids):
        lane = lanes[lane_id]

        nbr_ids = lane.predecessors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre_pairs.append([i, j])

        nbr_ids = lane.successors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc_pairs.append([i, j])

        nbr_id = lane.l_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                left_pairs.append([i, j])

        nbr_id = lane.r_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                right_pairs.append([i, j])
    pre_pairs = np.asarray(pre_pairs, np.int64)
    suc_pairs = np.asarray(suc_pairs, np.int64)
    left_pairs = np.asarray(left_pairs, np.int64)
    right_pairs = np.asarray(right_pairs, np.int64)

    graph = dict()
    graph['ctrs'] = np.concatenate(ctrs, 0)
    graph['num_nodes'] = num_nodes
    graph['feats'] = np.concatenate(feats, 0)
    graph['turn'] = np.concatenate(turn, 0)
    graph['control'] = np.concatenate(control, 0)
    graph['intersect'] = np.concatenate(intersect, 0)
    graph['pre'] = [pre]
    graph['suc'] = [suc]
    graph['lane_idcs'] = lane_idcs
    graph['pre_pairs'] = pre_pairs
    graph['suc_pairs'] = suc_pairs
    graph['left_pairs'] = left_pairs
    graph['right_pairs'] = right_pairs

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

    for key in ['pre', 'suc']:
        # config['num_scales'] = 6
        graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], 6)
    data['graph'] = graph        

    # 4. PreprocessDataset

    store = dict()
    for key in [
        "idx",
        "city",
        "feats",
        "ctrs",
        "orig",
        "theta",
        "rot",
        "gt_preds",
        "has_preds",
        "graph",
    ]:
        store[key] = to_numpy(data[key])
        if key in ["graph"]:
            store[key] = to_int16(store[key])

    graph = dict()
    for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
        graph[key] = ref_copy(store['graph'][key])

    # 5. Preprocess
    graph = from_numpy(graph)
    out = preprocess(to_long(gpu(graph)), 6, data=graph)

    store['graph']['left'] = out['left']
    store['graph']['right'] = out['right']
    
    return store, vids

    