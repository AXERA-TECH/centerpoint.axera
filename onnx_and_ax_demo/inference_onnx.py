#!/usr/bin/env python3
"""
CenterPoint ONNX Runtime Inference Demo.

Usage:
    python inference_onnx.py path/to/onnx_model/pointpillars.onnx path/to/extracted_data/config.json path/to/extracted_data \
        --output-dir path/to/inference_results --num-samples 50
"""

import argparse
import json
import os
import os.path as osp
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import numba


def parse_args():
    parser = argparse.ArgumentParser(description='CenterPoint ONNX Inference')
    parser.add_argument('onnx_model', help='ONNX model path')
    parser.add_argument('config_json', help='JSON config file path')
    parser.add_argument('data_dir', help='extracted data directory')
    parser.add_argument('--output-dir', default='./inference_results', help='output directory')
    parser.add_argument('--score-thr', type=float, default=0.1, help='score threshold')
    parser.add_argument('--device', default='cuda:0', help='device for ONNX inference')
    parser.add_argument('--num-samples', type=int, default=None, help='number of samples to process')
    parser.add_argument('--visualize', action='store_true', help='save visualization images and video')
    parser.add_argument('--fps', type=int, default=10, help='video fps')
    return parser.parse_args()


def load_onnx_model(onnx_path, device='cuda:0'):
    """Load ONNX model"""
    available_providers = ort.get_available_providers()
    
    providers = []
    if 'cuda' in device.lower() and 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    print(f"Loaded ONNX model from {onnx_path}")
    print(f"Using providers: {session.get_providers()}")
    return session


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_sample_index(data_dir):
    """Load sample index"""
    index_path = osp.join(data_dir, 'sample_index.json')
    with open(index_path, 'r') as f:
        sample_index = json.load(f)
    return sample_index


def load_points(data_dir, points_path):
    """Load point cloud data from binary file"""
    full_path = osp.join(data_dir, points_path)
    points = np.fromfile(full_path, dtype=np.float32).reshape(-1, 5)
    return points


def load_gt(data_dir, gt_path):
    """Load ground truth annotations"""
    full_path = osp.join(data_dir, gt_path)
    with open(full_path, 'r') as f:
        gt = json.load(f)
    return gt


@numba.jit(nopython=True)
def _points_to_voxel_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=20,
    max_voxels=30000,
):
    """Voxelization kernel using numba for acceleration"""
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points, voxel_size, coors_range, max_points=20, max_voxels=30000):
    """Convert point cloud to voxels
    
    Args:
        points: [N, 5] float32 array (x, y, z, intensity, time_lag)
        voxel_size: [3] voxel size (x, y, z)
        coors_range: [6] point cloud range (xmin, ymin, zmin, xmax, ymax, zmax)
        max_points: max points per voxel
        max_voxels: max number of voxels
        
    Returns:
        voxels: [M, max_points, 5] voxel features
        coors: [M, 3] voxel coordinates (z, y, x)
        num_points_per_voxel: [M] number of points in each voxel
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=np.float32)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=np.float32)
    
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # reverse to (z, y, x)
    
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(shape=(max_voxels, max_points, points.shape[-1]), dtype=np.float32)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    
    voxel_num = _points_to_voxel_kernel(
        points.astype(np.float32),
        voxel_size,
        coors_range,
        num_points_per_voxel,
        coor_to_voxelidx,
        voxels,
        coors,
        max_points,
        max_voxels,
    )
    
    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    
    return voxels, coors, num_points_per_voxel


def preprocess_pointpillars(points, config):
    """Preprocess point cloud for PointPillars model
    
    This function converts raw point cloud to the input format required by the ONNX model.
    """
    voxel_cfg = config['voxel_generator']
    voxel_size = np.array(voxel_cfg['voxel_size'], dtype=np.float32)
    pc_range = np.array(voxel_cfg['range'], dtype=np.float32)
    max_points = voxel_cfg['max_points_in_voxel']
    max_voxels = voxel_cfg['max_voxel_num'][1] if isinstance(voxel_cfg['max_voxel_num'], list) else voxel_cfg['max_voxel_num']
    
    # Voxelization
    voxels, coors, num_points = points_to_voxel(
        points, voxel_size, pc_range, max_points, max_voxels
    )
    
    return voxels, coors, num_points


@numba.jit(nopython=True)
def _create_pillars_input_kernel(voxels, coors, num_points, features, indices,
                                  voxel_size, pc_range, bev_w, num_voxels):
    """Numba-accelerated kernel for pillar feature computation"""
    for i in range(num_voxels):
        n_points = num_points[i]
        if n_points == 0:
            continue
        
        voxel = voxels[i]
        coor = coors[i]
        
        # Compute pillar center (vectorized sum)
        x_sum = 0.0
        y_sum = 0.0
        z_sum = 0.0
        for j in range(n_points):
            x_sum += voxel[j, 0]
            y_sum += voxel[j, 1]
            z_sum += voxel[j, 2]
        x_center = x_sum / n_points
        y_center = y_sum / n_points
        z_center = z_sum / n_points
        
        # Compute pillar position
        x_pillar = coor[2] * voxel_size[0] + pc_range[0] + voxel_size[0] / 2
        y_pillar = coor[1] * voxel_size[1] + pc_range[1] + voxel_size[1] / 2
        
        # Fill features
        for j in range(n_points):
            features[0, i, j] = voxel[j, 0]  # x
            features[1, i, j] = voxel[j, 1]  # y
            features[2, i, j] = voxel[j, 2]  # z
            features[3, i, j] = voxel[j, 3]  # intensity
            features[4, i, j] = voxel[j, 4]  # time_lag
            features[5, i, j] = voxel[j, 0] - x_center  # x_c
            features[6, i, j] = voxel[j, 1] - y_center  # y_c
            features[7, i, j] = voxel[j, 2] - z_center  # z_c
            features[8, i, j] = voxel[j, 0] - x_pillar  # x_p
            features[9, i, j] = voxel[j, 1] - y_pillar  # y_p
        
        # Compute BEV index
        indices[i, 1] = coor[1] * bev_w + coor[2]


def create_pillars_input(voxels, coors, num_points, config, max_pillars=30000):
    """Create input tensors for the PointPillars ONNX model (numba-accelerated)
    
    The model expects:
    - features: [1, 10, max_pillars, max_points_per_pillar]
    - indices: [1, max_pillars, 2]
    """
    voxel_cfg = config['voxel_generator']
    voxel_size = np.array(voxel_cfg['voxel_size'], dtype=np.float32)
    pc_range = np.array(voxel_cfg['range'], dtype=np.float32)
    max_points_per_pillar = voxel_cfg['max_points_in_voxel']
    
    num_voxels = voxels.shape[0]
    
    # Pad or truncate to max_pillars
    if num_voxels > max_pillars:
        voxels = voxels[:max_pillars]
        coors = coors[:max_pillars]
        num_points = num_points[:max_pillars]
        num_voxels = max_pillars
    
    # Initialize tensors
    features = np.zeros((10, max_pillars, max_points_per_pillar), dtype=np.float32)
    indices = np.zeros((max_pillars, 2), dtype=np.int64)  # ONNX needs int64
    indices[:, 0] = 0  # batch index
    indices[:, 1] = -1  # invalid index marker
    
    # BEV grid size
    bev_w = int((pc_range[3] - pc_range[0]) / voxel_size[0])
    
    # Call numba kernel
    _create_pillars_input_kernel(
        voxels, coors, num_points, features, indices,
        voxel_size, pc_range, bev_w, num_voxels
    )
    
    # Add batch dimension
    features = features[np.newaxis, ...]  # [1, 10, max_pillars, max_points_per_pillar]
    indices = indices[np.newaxis, ...]    # [1, max_pillars, 2]
    
    return features, indices


def decode_bbox(reg, height, dim, rot, vel, score, cls, config, task_idx):
    """Decode detection outputs to 3D bounding boxes
    
    Args:
        reg: [H, W, 2] registration offset
        height: [H, W, 1] height
        dim: [H, W, 3] dimensions (l, h, w)
        rot: [H, W, 2] rotation (sin, cos)
        vel: [H, W, 2] velocity
        score: [H, W] confidence score
        cls: [H, W] class prediction
        config: configuration dict
        task_idx: task index for class offset
        
    Returns:
        boxes: [N, 9] (x, y, z, w, l, h, theta, vx, vy)
        scores: [N]
        labels: [N]
    """
    test_cfg = config['test_cfg']
    voxel_size = test_cfg['voxel_size']
    pc_range = test_cfg['pc_range']
    out_size_factor = test_cfg['out_size_factor']
    score_threshold = test_cfg['score_threshold']
    
    H, W = score.shape
    
    # Create grid
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    
    # Decode center
    xs = (xs + reg[..., 0]) * out_size_factor * voxel_size[0] + pc_range[0]
    ys = (ys + reg[..., 1]) * out_size_factor * voxel_size[1] + pc_range[1]
    zs = height[..., 0]
    
    # Decode rotation
    theta = np.arctan2(rot[..., 0], rot[..., 1])
    
    # Get class offset for this task
    class_offset = [0, 1, 3, 5, 6, 8][task_idx]
    
    # Filter by score
    mask = score > score_threshold
    
    if not np.any(mask):
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,)), np.zeros((0,), dtype=np.int32)
    
    # Extract valid predictions
    xs = xs[mask]
    ys = ys[mask]
    zs = zs[mask]
    dims = dim[mask]  # [N, 3] (l, h, w)
    theta = theta[mask]
    vels = vel[mask]  # [N, 2]
    scores = score[mask]
    labels = cls[mask] + class_offset
    
    # Construct boxes: [x, y, z, w, l, h, theta, vx, vy]
    boxes = np.stack([
        xs, ys, zs,
        dims[:, 2],  # w
        dims[:, 0],  # l
        dims[:, 1],  # h
        theta,
        vels[:, 0],  # vx
        vels[:, 1],  # vy
    ], axis=-1)
    
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int32)


@numba.jit(nopython=True)
def _nms_bev_kernel(boxes, scores, nms_threshold, max_output=500):
    """Numba-accelerated NMS kernel"""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    
    # Sort by score descending
    order = np.argsort(-scores)
    
    # Pre-compute box corners
    x1 = boxes[:, 0] - boxes[:, 4] / 2  # x - l/2
    y1 = boxes[:, 1] - boxes[:, 3] / 2  # y - w/2
    x2 = boxes[:, 0] + boxes[:, 4] / 2  # x + l/2
    y2 = boxes[:, 1] + boxes[:, 3] / 2  # y + w/2
    areas = boxes[:, 3] * boxes[:, 4]  # w * l
    
    suppressed = np.zeros(n, dtype=np.int32)
    keep = np.zeros(max_output, dtype=np.int64)
    num_keep = 0
    
    for _i in range(n):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        
        keep[num_keep] = i
        num_keep += 1
        if num_keep >= max_output:
            break
        
        # Compute IoU with remaining boxes
        for _j in range(_i + 1, n):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            
            # Compute intersection
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            
            # Compute IoU
            union = areas[i] + areas[j] - inter
            iou = inter / max(union, 1e-6)
            
            if iou > nms_threshold:
                suppressed[j] = 1
    
    return keep[:num_keep]


def nms_bev(boxes, scores, labels, nms_threshold=0.2):
    """Aligned BEV NMS (numba-accelerated)"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    return _nms_bev_kernel(boxes, scores, nms_threshold)


def postprocess(outputs, config, score_thr=0.1):
    """Postprocess model outputs
    
    CenterPoint model output structure (42 outputs total, 7 per task, 6 tasks):
    Per task output order:
      - reg: [1, 2, 128, 128] - registration offset
      - height: [1, 1, 128, 128] - height
      - dim: [1, 3, 128, 128] - dimensions (l, h, w)
      - rot: [1, 2, 128, 128] - rotation (sin, cos)
      - vel: [1, 2, 128, 128] - velocity
      - score: [1, 128, 128] - confidence (after sigmoid)
      - cls: [1, 128, 128] - class index (after argmax)
    
    Args:
        outputs: model outputs (list of 42 tensors)
        config: configuration dict
        score_thr: score threshold
        
    Returns:
        boxes: [N, 9] (x, y, z, w, l, h, theta, vx, vy)
        scores: [N]
        labels: [N]
    """
    tasks = config['tasks']
    num_tasks = len(tasks)  # 6 tasks
    outputs_per_task = 7  # reg, height, dim, rot, vel, score, cls
    
    test_cfg = config['test_cfg']
    voxel_size = test_cfg['voxel_size']
    pc_range = test_cfg['pc_range']
    out_size_factor = test_cfg['out_size_factor']
    score_threshold = test_cfg['score_threshold']
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # Class offset for each task
    class_offsets = [0, 1, 3, 5, 6, 8]  # car, truck/constr, bus/trailer, barrier, moto/bicycle, ped/cone
    
    for task_idx in range(num_tasks):
        base_idx = task_idx * outputs_per_task
        
        # Extract outputs for this task
        reg = outputs[base_idx + 0][0]      # [2, H, W]
        height = outputs[base_idx + 1][0]   # [1, H, W]
        dim = outputs[base_idx + 2][0]      # [3, H, W]
        rot = outputs[base_idx + 3][0]      # [2, H, W]
        vel = outputs[base_idx + 4][0]      # [2, H, W]
        score = outputs[base_idx + 5][0]    # [H, W] - already after sigmoid
        cls = outputs[base_idx + 6][0]      # [H, W] - class index
        
        H, W = score.shape
        
        # Create coordinate grid
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        xs, ys = np.meshgrid(xs, ys)
        
        # Decode center position
        # reg is [2, H, W], reg[0] is x offset, reg[1] is y offset
        center_x = (xs + reg[0]) * out_size_factor * voxel_size[0] + pc_range[0]
        center_y = (ys + reg[1]) * out_size_factor * voxel_size[1] + pc_range[1]
        center_z = height[0]  # [H, W]
        
        # Decode dimensions
        # dim  [3, H, W], order: l, h, w
        dim_l = dim[0]  # length
        dim_h = dim[1]  # height  
        dim_w = dim[2]  # width
        
        # Decode rotation
        # rot is [2, H, W], rot[0] is sin, rot[1] is cos
        theta = np.arctan2(rot[0], rot[1])
        
        # Velocity
        vel_x = vel[0]
        vel_y = vel[1]
        
        # Filter by score threshold
        mask = score > score_threshold
        
        if not np.any(mask):
            continue
        
        # Get class offset for this task
        class_offset = class_offsets[task_idx]
        
        # Extract valid predictions
        boxes = np.stack([
            center_x[mask],
            center_y[mask],
            center_z[mask],
            dim_w[mask],   # w
            dim_l[mask],   # l
            dim_h[mask],   # h
            theta[mask],
            vel_x[mask],
            vel_y[mask],
        ], axis=-1).astype(np.float32)
        
        scores_task = score[mask].astype(np.float32)
        labels_task = (cls[mask] + class_offset).astype(np.int32)
        
        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_scores.append(scores_task)
            all_labels.append(labels_task)
    
    if len(all_boxes) == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,)), np.zeros((0,), dtype=np.int32)
    
    # Concatenate all detections
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Apply NMS
    nms_cfg = config['test_cfg']['nms']
    keep = nms_bev(boxes, scores, labels, nms_cfg['nms_iou_threshold'])
    
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Filter by final score threshold
    mask = scores > score_thr
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Limit max detections
    max_per_img = config['test_cfg']['max_per_img']
    if len(boxes) > max_per_img:
        topk_indices = np.argsort(-scores)[:max_per_img]
        boxes = boxes[topk_indices]
        scores = scores[topk_indices]
        labels = labels[topk_indices]
    
    return boxes, scores, labels


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# BGR colors
CLASS_COLORS_BGR = {
    0: (255, 0, 0),      # car - blue
    1: (0, 165, 255),    # truck - orange
    2: (0, 0, 255),      # construction_vehicle - red
    3: (0, 255, 255),    # bus - yellow
    4: (128, 0, 128),    # trailer - purple
    5: (255, 255, 0),    # barrier - cyan
    6: (0, 0, 255),      # motorcycle - red
    7: (0, 255, 0),      # bicycle - green
    8: (255, 0, 255),    # pedestrian - magenta
    9: (0, 255, 255),    # traffic_cone - yellow
}


def visualize_bev(points, boxes, scores, labels, config, save_path, 
                  frame_idx=0, eval_range=35, conf_th=0.5):
    """BEV visualization using OpenCV """
    try:
        import cv2
    except ImportError:
        print("opencv-python not available, skipping visualization")
        return None
    
    # Image size and scale
    img_size = 800
    scale = img_size / (2 * eval_range)
    center = img_size // 2
    
    # Create black background
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Filter points within range
    mask = (np.abs(points[:, 0]) < eval_range) & (np.abs(points[:, 1]) < eval_range)
    pts = points[mask, :3]
    
    # Remove close points
    close_mask = (np.abs(pts[:, 0]) < 3) & (np.abs(pts[:, 1]) < 3)
    pts = pts[~close_mask]
    
    # Calculate distances for coloring (viridis-like: purple->cyan->yellow)
    dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    norm_dists = np.minimum(1.0, dists / eval_range)
    
    # Convert to image coordinates and draw points
    px = (center + pts[:, 0] * scale).astype(np.int32)
    py = (center - pts[:, 1] * scale).astype(np.int32)
    
    # Filter valid points (within image bounds)
    valid = (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    px, py, norm_dists = px[valid], py[valid], norm_dists[valid]
    
    # Viridis-like colormap using vectorized operations
    t = norm_dists
    r = np.where(t < 0.5, 68 + t * 2 * (49 - 68), 49 + (t - 0.5) * 2 * (253 - 49))
    g = np.where(t < 0.5, 1 + t * 2 * (104 - 1), 104 + (t - 0.5) * 2 * (231 - 104))
    b = np.where(t < 0.5, 84 + t * 2 * (142 - 84), 142 + (t - 0.5) * 2 * (37 - 142))
    
    # Draw all points at once
    img[py, px, 0] = b.astype(np.uint8)
    img[py, px, 1] = g.astype(np.uint8)
    img[py, px, 2] = r.astype(np.uint8)
    
    # Count detections
    num_detections = sum(1 for s in scores if s >= conf_th)
    
    # Draw detection boxes with class-specific shapes
    for box, score, label in zip(boxes, scores, labels):
        if score < conf_th:
            continue
        
        x, y, z, w, l, h, theta, vx, vy = box
        label_int = int(label)
        
        # Get color for this class
        color = CLASS_COLORS_BGR.get(label_int, (255, 255, 255))
        
        # Convert center to image coordinates
        cx = int(center + x * scale)
        cy = int(center - y * scale)
        
        # Apply angle transformation (same as demo_utils)
        vis_theta = -theta - np.pi / 2
        cos_t, sin_t = np.cos(vis_theta), np.sin(vis_theta)
        
        # Different shapes based on class
        if label_int == 8:  # pedestrian - circle
            radius = max(3, int(max(w, l) * scale / 2))
            cv2.circle(img, (cx, cy), radius, color, 2)
            # Draw heading line
            head_x = int(cx + radius * cos_t)
            head_y = int(cy - radius * sin_t)
            cv2.line(img, (cx, cy), (head_x, head_y), color, 2)
            
        elif label_int == 9:  # traffic_cone - small triangle
            size = max(4, int(max(w, l) * scale))
            pts = np.array([
                [cx, cy - size],  # top
                [cx - size//2, cy + size//2],  # bottom left
                [cx + size//2, cy + size//2],  # bottom right
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
            
        elif label_int == 5:  # barrier - thin rectangle
            # Box corners (thin barrier)
            corners = np.array([
                [l/2, w/4], [l/2, -w/4], [-l/2, -w/4], [-l/2, w/4]
            ])
            rot_corners = np.zeros_like(corners)
            rot_corners[:, 0] = corners[:, 0] * cos_t - corners[:, 1] * sin_t + x
            rot_corners[:, 1] = corners[:, 0] * sin_t + corners[:, 1] * cos_t + y
            corners_img = np.zeros((4, 2), dtype=np.int32)
            corners_img[:, 0] = (center + rot_corners[:, 0] * scale).astype(np.int32)
            corners_img[:, 1] = (center - rot_corners[:, 1] * scale).astype(np.int32)
            cv2.fillPoly(img, [corners_img], color)
            
        elif label_int in [6, 7]:  # motorcycle, bicycle - small box with direction
            # Smaller box for bikes
            corners = np.array([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            rot_corners = np.zeros_like(corners)
            rot_corners[:, 0] = corners[:, 0] * cos_t - corners[:, 1] * sin_t + x
            rot_corners[:, 1] = corners[:, 0] * sin_t + corners[:, 1] * cos_t + y
            corners_img = np.zeros((4, 2), dtype=np.int32)
            corners_img[:, 0] = (center + rot_corners[:, 0] * scale).astype(np.int32)
            corners_img[:, 1] = (center - rot_corners[:, 1] * scale).astype(np.int32)
            cv2.polylines(img, [corners_img], True, color, 2)
            # Draw prominent heading arrow
            front_mid = ((corners_img[0] + corners_img[1]) // 2).astype(np.int32)
            cv2.arrowedLine(img, (cx, cy), tuple(front_mid), color, 2, tipLength=0.4)
            
        else:  # car, truck, bus, trailer, construction_vehicle - standard box
            # Box corners
            corners = np.array([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            rot_corners = np.zeros_like(corners)
            rot_corners[:, 0] = corners[:, 0] * cos_t - corners[:, 1] * sin_t + x
            rot_corners[:, 1] = corners[:, 0] * sin_t + corners[:, 1] * cos_t + y
            corners_img = np.zeros((4, 2), dtype=np.int32)
            corners_img[:, 0] = (center + rot_corners[:, 0] * scale).astype(np.int32)
            corners_img[:, 1] = (center - rot_corners[:, 1] * scale).astype(np.int32)
            cv2.polylines(img, [corners_img], True, color, 2)
            # Draw front indicator line
            front_mid = ((corners_img[0] + corners_img[1]) // 2).astype(np.int32)
            cv2.line(img, (cx, cy), tuple(front_mid), color, 2)
    
    # Draw frame info (white text)
    cv2.putText(img, f'Frame: {frame_idx}', (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f'Detections: {num_detections}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw legend
    legend_y = 80
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        cv2.rectangle(img, (10, legend_y), (25, legend_y + 12), color, -1)
        cv2.putText(img, cls_name, (30, legend_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 18
    
    # Save image
    cv2.imwrite(save_path, img)
    return True


def create_video_from_images(image_dir, output_video_path, fps=10):
    """Create video from images in a directory
    
    Args:
        image_dir: directory containing images
        output_video_path: output video file path
        fps: frames per second
    """
    try:
        import cv2
    except ImportError:
        print("opencv-python not available, cannot create video")
        return
    
    # Get all image files sorted by name
    image_files = sorted([f for f in os.listdir(image_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(osp.join(image_dir, image_files[0]))
    if first_img is None:
        print(f"Cannot read first image: {image_files[0]}")
        return
    
    height, width = first_img.shape[:2]
    
    # Limit video size for better compatibility
    max_width, max_height = 1920, 1080
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        width, height = int(width * scale), int(height * scale)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = output_video_path.replace('.mp4', '.avi')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {output_video_path}")
    
    for img_file in tqdm(image_files, desc="Creating video"):
        img_path = osp.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            video_writer.write(img)
    
    video_writer.release()
    print(f"Video saved to {output_video_path}")


def run_inference(session, points, config):
    """Run inference on a single point cloud
    
    Args:
        session: ONNX Runtime session
        points: [N, 5] point cloud
        config: configuration dict
        
    Returns:
        boxes: [M, 9] detected boxes
        scores: [M] detection scores
        labels: [M] class labels
    """
    # Preprocess
    voxels, coors, num_points = preprocess_pointpillars(points, config)
    
    # Create model input
    features, indices = create_pillars_input(voxels, coors, num_points, config)
    
    # Get input names
    input_names = [inp.name for inp in session.get_inputs()]
    
    # Build feed dict based on exact input names
    # Model expects: input.1 (features) and indices_input (indices)
    feed_dict = {}
    for name in input_names:
        if name == 'input.1':
            feed_dict[name] = features.astype(np.float32)
        elif name == 'indices_input':
            feed_dict[name] = indices.astype(np.int64)
        elif 'indices' in name.lower():
            feed_dict[name] = indices.astype(np.int64)
        else:
            feed_dict[name] = features.astype(np.float32)
    
    # Run inference
    outputs = session.run(None, feed_dict)
    
    # Postprocess
    boxes, scores, labels = postprocess(outputs, config)
    
    return boxes, scores, labels


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config_json)
    print(f"Loaded config from {args.config_json}")
    
    # Load ONNX model
    session = load_onnx_model(args.onnx_model, args.device)
    
    # Print model info
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")
    
    # Load sample index
    sample_index = load_sample_index(args.data_dir)
    samples = sample_index['samples']
    
    # Limit samples if specified
    if args.num_samples is not None:
        samples = samples[:args.num_samples]
    
    print(f"\nProcessing {len(samples)} samples...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create images directory for visualization
    images_dir = osp.join(args.output_dir, 'images')
    if args.visualize:
        os.makedirs(images_dir, exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Process each sample
    for idx, sample in enumerate(tqdm(samples, desc="Inference")):
        token = sample['token']
        
        # Load point cloud
        points = load_points(args.data_dir, sample['points_path'])
        
        # Run inference
        boxes, scores, labels = run_inference(session, points, config)
        
        # Store results
        result = {
            'token': token,
            'boxes': boxes.tolist(),
            'scores': scores.tolist(),
            'labels': labels.tolist(),
            'num_detections': len(boxes),
        }
        all_results.append(result)
        
        # Visualize if requested
        if args.visualize:
            vis_path = osp.join(images_dir, f'frame_{idx:06d}.png')
            visualize_bev(points, boxes, scores, labels, config, vis_path, frame_idx=idx)
    
    # Save results
    results_path = osp.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Create video from images
    if args.visualize:
        video_path = osp.join(args.output_dir, 'centerpoint_detection_onnx.mp4')
        create_video_from_images(images_dir, video_path, fps=args.fps)
    
    # Print summary
    total_detections = sum(r['num_detections'] for r in all_results)
    print(f"Total detections: {total_detections}")
    print(f"Average detections per sample: {total_detections / len(samples):.1f}")


if __name__ == '__main__':
    main()
