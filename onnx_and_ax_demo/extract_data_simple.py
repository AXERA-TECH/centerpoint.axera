#!/usr/bin/env python3
"""
Simple data extraction script that reads nuScenes point cloud data directly.
Usage:
    python extract_data_simple.py --data-root path/to/data --output-dir ./extracted_data --num-samples 50
    the structure of data should be like this:
    data
    ├── samples
    │   ├── LIDAR_TOP
    │   │   ├── 000000.bin
    │   │   ├── 000001.bin
    │   │   └── ...
    └── ...
    the data is in the LIDAR_TOP folder, and the files are named like 000000.bin, 000001.bin, etc.
"""

import argparse
import json
import os
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Extract nuScenes LiDAR data')
    parser.add_argument('--data-root', default='../data', help='nuScenes data root')
    parser.add_argument('--output-dir', default='./extracted_data', help='output directory')
    parser.add_argument('--num-samples', type=int, default=20, help='number of samples to extract')
    return parser.parse_args()


def load_nusc_lidar(lidar_path):
    """Load nuScenes LiDAR point cloud
    
    nuScenes LiDAR format: [x, y, z, intensity, ring_index] (5 floats per point)
    """
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    # Replace ring_index with time_lag (0 for single sweep)
    points[:, 4] = 0.0
    return points


def save_config(output_dir):
    """Save default CenterPoint configuration"""
    config = {
        'model': {
            'type': 'PointPillars',
            'voxel_size': [0.2, 0.2, 8.0],
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'num_input_features': 5,
        },
        'voxel_generator': {
            'range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'voxel_size': [0.2, 0.2, 8.0],
            'max_points_in_voxel': 20,
            'max_voxel_num': [30000, 60000],
        },
        'test_cfg': {
            'post_center_limit_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            'max_per_img': 500,
            'nms': {
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 83,
                'nms_iou_threshold': 0.2,
            },
            'score_threshold': 0.1,
            'pc_range': [-51.2, -51.2],
            'out_size_factor': 4,
            'voxel_size': [0.2, 0.2],
        },
        'class_names': [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        'tasks': [
            {'num_class': 1, 'class_names': ['car']},
            {'num_class': 2, 'class_names': ['truck', 'construction_vehicle']},
            {'num_class': 2, 'class_names': ['bus', 'trailer']},
            {'num_class': 1, 'class_names': ['barrier']},
            {'num_class': 2, 'class_names': ['motorcycle', 'bicycle']},
            {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']},
        ],
    }
    
    config_path = osp.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    return config


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    points_dir = osp.join(args.output_dir, 'points')
    gt_dir = osp.join(args.output_dir, 'gt_annotations')
    os.makedirs(points_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    save_config(args.output_dir)
    
    lidar_dir = osp.join(args.data_root, 'samples', 'LIDAR_TOP')
    if not osp.exists(lidar_dir):
        print(f"Error: LiDAR directory not found: {lidar_dir}")
        return
    
    lidar_files = sorted(glob(osp.join(lidar_dir, '*.bin')))
    print(f"Found {len(lidar_files)} LiDAR files")
    
    lidar_files = lidar_files[:args.num_samples]
    
    sample_index = {
        'samples': [],
        'total_samples': len(lidar_files),
    }
    

    print(f"Extracting {len(lidar_files)} samples...")
    for i, lidar_path in enumerate(tqdm(lidar_files, desc="Extracting")):
        filename = osp.basename(lidar_path)
        token = filename.replace('.pcd.bin', '').replace('.bin', '')
        
        points = load_nusc_lidar(lidar_path)
        
        out_path = osp.join(points_dir, f'{token}.bin')
        points.astype(np.float32).tofile(out_path)
        
        gt_info = {
            'token': token,
            'gt_boxes': [],
            'gt_names': [],
        }
        gt_path = osp.join(gt_dir, f'{token}.json')
        with open(gt_path, 'w') as f:
            json.dump(gt_info, f, indent=2)
    
        sample_index['samples'].append({
            'token': token,
            'points_path': f'points/{token}.bin',
            'gt_path': f'gt_annotations/{token}.json',
            'num_points': points.shape[0],
        })

    index_path = osp.join(args.output_dir, 'sample_index.json')
    with open(index_path, 'w') as f:
        json.dump(sample_index, f, indent=2)
    
    print(f"\nDone!")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Total samples: {len(lidar_files)}")
    print(f"\nTo run inference:")
    print(f"  python inference_onnx.py ../onnx_model/pointpillars.onnx {args.output_dir}/config.json {args.output_dir} --visualize")


if __name__ == '__main__':
    main()
