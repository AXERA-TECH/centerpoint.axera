#!/usr/bin/env python3
#
# Prepare calibration dataset for CenterPoint NPU quantization.
# This script generates features.tar.gz and indices.tar.gz from point cloud data.
#
# Usage (two modes):
#
# Mode 1: From raw dataset (requires pickle info file)
#   python prepare_calib_data.py --data_root path/to/data --output_dir path/to/output --num_samples 32
#   The structure of data should be like this:
#   data
#   ├── samples
#   │   ├── LIDAR_TOP
#   │   │   ├── 000000.bin
#   │   │   ├── 000001.bin
#   │   │   └── ...
#   ├── infos_val_10sweeps_withvelo_filter_True.pkl
#   └── ...
#
# Mode 2: From existing points bin files directory (simpler, recommended)
#   python prepare_calib_data.py --from_points_dir path/to/points --output_dir path/to/output --num_samples 32
#   The points directory should contain .bin files:
#   points/
#   ├── 000000.bin
#   ├── 000001.bin
#   └── ...
#   (This is useful when you already have extracted point cloud files, e.g., from extract_data_simple.py)
#
# Output:
#   - features.tar.gz: Contains sample_XXXX.npy files, each with shape (1, 10, 30000, 20)
#   - indices.tar.gz: Contains sample_XXXX.npy files, each with shape (1, 30000, 2)

import os
import sys
import argparse
import tarfile
import numpy as np
from pathlib import Path

CENTERPOINT_ROOT = Path(__file__).parent
sys.path.insert(0, str(CENTERPOINT_ROOT))

X_STEP = 0.2
Y_STEP = 0.2
X_MIN = -51.2
X_MAX = 51.2
Y_MIN = -51.2
Y_MAX = 51.2
Z_MIN = -5.0
Z_MAX = 3.0
BEV_W = 512
BEV_H = 512
MAX_PILLARS = 30000
MAX_POINTS_IN_PILLARS = 20
FEATURE_NUM = 10


def preprocess_point_cloud(points: np.ndarray) -> tuple:
    """
    Convert raw point cloud to the input format required by the ONNX model
    
    Args:
        points: raw point cloud data, shape (N, 5) - [x, y, z, intensity, time_lag]
    
    Returns:
        features: shape (1, FEATURE_NUM, MAX_PILLARS, MAX_POINTS_IN_PILLARS)
        indices: shape (1, MAX_PILLARS, 2)
    """
    # Initialize output
    features = np.zeros((1, FEATURE_NUM, MAX_PILLARS, MAX_POINTS_IN_PILLARS), dtype=np.float32)
    indices = np.zeros((1, MAX_PILLARS, 2), dtype=np.int64)
    
    # Filter out points that are out of range
    mask = (
        (points[:, 0] >= X_MIN) & (points[:, 0] < X_MAX) &
        (points[:, 1] >= Y_MIN) & (points[:, 1] < Y_MAX) &
        (points[:, 2] >= Z_MIN) & (points[:, 2] < Z_MAX)
    )
    points = points[mask]
    
    # Calculate the index of the pillar that each point belongs to
    x_idx = ((points[:, 0] - X_MIN) / X_STEP).astype(np.int32)
    y_idx = ((points[:, 1] - Y_MIN) / Y_STEP).astype(np.int32)
    
    # Ensure the index is within the valid range
    x_idx = np.clip(x_idx, 0, BEV_W - 1)
    y_idx = np.clip(y_idx, 0, BEV_H - 1)
    
    # 2D index of the pillar
    pillar_2d_idx = y_idx * BEV_W + x_idx
    
    # Find all unique pillars
    unique_pillars, inverse_indices = np.unique(pillar_2d_idx, return_inverse=True)
    num_pillars = min(len(unique_pillars), MAX_PILLARS)
    
    # Assign a sequential index to each pillar
    pillar_to_idx = {pillar: idx for idx, pillar in enumerate(unique_pillars[:num_pillars])}
    
    # Count the number of points in each pillar
    point_count_per_pillar = np.zeros(num_pillars, dtype=np.int32)
    
    # Temporarily store the points in each pillar
    pillar_points = [[] for _ in range(num_pillars)]
    
    for i, point in enumerate(points):
        pillar = pillar_2d_idx[i]
        if pillar not in pillar_to_idx:
            continue
        pillar_idx = pillar_to_idx[pillar]
        if point_count_per_pillar[pillar_idx] < MAX_POINTS_IN_PILLARS:
            pillar_points[pillar_idx].append(point)
            point_count_per_pillar[pillar_idx] += 1
    
    # Fill the features and indices
    for pillar_idx, pillar_2d in enumerate(unique_pillars[:num_pillars]):
        pts = pillar_points[pillar_idx]
        if len(pts) == 0:
            continue
        
        pts = np.array(pts)
        num_pts = len(pts)
        
        # Calculate the center coordinates of the pillar
        x_idx_val = int(pillar_2d % BEV_W)
        y_idx_val = int(pillar_2d // BEV_W)
        pillar_center_x = x_idx_val * X_STEP + X_MIN + X_STEP / 2
        pillar_center_y = y_idx_val * Y_STEP + Y_MIN + Y_STEP / 2
        
        # Calculate the center of the point cloud (cluster center)
        x_center = pts[:, 0].mean()
        y_center = pts[:, 1].mean()
        z_center = pts[:, 2].mean()
        
        # Fill the features
        for pt_idx, pt in enumerate(pts):
            # Basic features: x, y, z, intensity, time_lag
            features[0, 0, pillar_idx, pt_idx] = pt[0]  # x
            features[0, 1, pillar_idx, pt_idx] = pt[1]  # y
            features[0, 2, pillar_idx, pt_idx] = pt[2]  # z
            features[0, 3, pillar_idx, pt_idx] = pt[3]  # intensity
            features[0, 4, pillar_idx, pt_idx] = pt[4] if len(pt) > 4 else 0  # time_lag
            
            # Offset relative to the cluster center
            features[0, 5, pillar_idx, pt_idx] = pt[0] - x_center
            features[0, 6, pillar_idx, pt_idx] = pt[1] - y_center
            features[0, 7, pillar_idx, pt_idx] = pt[2] - z_center
            
            # Offset relative to the pillar center
            features[0, 8, pillar_idx, pt_idx] = pt[0] - pillar_center_x
            features[0, 9, pillar_idx, pt_idx] = pt[1] - pillar_center_y
        
        # Fill the indices: [batch_id, pillar_2d_idx]
        indices[0, pillar_idx, 0] = 0  # batch_id (always 0 for single batch)
        indices[0, pillar_idx, 1] = int(pillar_2d)
    
    return features, indices


def load_points_from_bin(bin_path: str) -> np.ndarray:
    """Load point cloud data from bin file"""
    points = np.fromfile(bin_path, dtype=np.float32)
    # nuScenes point cloud format: x, y, z, intensity, ring_index (5 features)
    # But CenterPoint uses: x, y, z, intensity, time_lag
    points = points.reshape(-1, 5)
    return points


def prepare_calibration_dataset(
    data_root: str,
    output_dir: str,
    num_samples: int = 32,
    use_mini: bool = True
):
    """
    Prepare the calibration dataset for quantization
    
    Args:
        data_root: CenterPoint data root directory (contains data directory)
        output_dir: Output directory
        num_samples: Number of samples
        use_mini: Whether to use mini dataset
    """
    import pickle
    
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    indices_dir = output_dir / "indices"
    
    # Create output directories
    features_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset information
    data_path = Path(data_root) / "data"
    info_file = data_path / "infos_val_10sweeps_withvelo_filter_True.pkl"
    
    if not info_file.exists():
        print(f"Error: Info file not found: {info_file}")
        print("Trying alternative path...")
        info_file = data_path / "nuscenes_infos_temporal_val.pkl"
    
    if not info_file.exists():
        raise FileNotFoundError(f"Cannot find info file in {data_path}")
    
    print(f"Loading info file: {info_file}")
    with open(info_file, 'rb') as f:
        infos = pickle.load(f)
    
    # Process pickle file format
    if isinstance(infos, dict) and 'infos' in infos:
        infos = infos['infos']
    
    print(f"Total samples in dataset: {len(infos)}")
    
    # Sampling
    sample_indices = np.linspace(0, len(infos) - 1, num_samples, dtype=np.int64)
    
    saved_count = 0
    for i, idx in enumerate(sample_indices):
        info = infos[idx]
        
        # Get lidar path
        lidar_path = info.get('lidar_path', None)
        if lidar_path is None:
            # Try other possible key names
            if 'sweeps' in info and len(info['sweeps']) > 0:
                lidar_path = info['sweeps'][0].get('lidar_path', None)
        
        if lidar_path is None:
            print(f"Warning: Cannot find lidar path for sample {idx}, skipping...")
            continue
        
        # Build full path
        if not os.path.isabs(lidar_path):
            lidar_path = os.path.join(data_root, lidar_path)
        
        # If the lidar path is relative to the data directory
        if not os.path.exists(lidar_path):
            lidar_path = os.path.join(data_path, os.path.basename(lidar_path))
        
        if not os.path.exists(lidar_path):
            # Try to find in the samples directory
            possible_paths = [
                data_path / "samples" / "LIDAR_TOP" / os.path.basename(lidar_path),
                Path(data_root) / "samples" / "LIDAR_TOP" / os.path.basename(lidar_path),
            ]
            for p in possible_paths:
                if p.exists():
                    lidar_path = str(p)
                    break
        
        if not os.path.exists(lidar_path):
            print(f"Warning: Lidar file not found: {lidar_path}, skipping...")
            continue
        
        print(f"Processing sample {i+1}/{num_samples}: {os.path.basename(lidar_path)}")
        
        # Load point cloud
        points = load_points_from_bin(lidar_path)
        
        # Preprocess
        features, indices = preprocess_point_cloud(points)
        
        # Save as numpy file
        np.save(features_dir / f"sample_{saved_count:04d}.npy", features)
        np.save(indices_dir / f"sample_{saved_count:04d}.npy", indices)
        
        saved_count += 1
        
        if saved_count >= num_samples:
            break
    
    print(f"\nSaved {saved_count} samples")
    
    # Pack into tar.gz
    print("\nCreating tar.gz archives...")
    
    features_tar_path = output_dir / "features.tar.gz"
    with tarfile.open(features_tar_path, "w:gz") as tar:
        for npy_file in sorted(features_dir.glob("*.npy")):
            tar.add(npy_file, arcname=npy_file.name)
    print(f"Created: {features_tar_path}")
    
    indices_tar_path = output_dir / "indices.tar.gz"
    with tarfile.open(indices_tar_path, "w:gz") as tar:
        for npy_file in sorted(indices_dir.glob("*.npy")):
            tar.add(npy_file, arcname=npy_file.name)
    print(f"Created: {indices_tar_path}")
    
    # Print input shape information
    print("\n" + "="*50)
    print("Calibration dataset preparation completed!")
    print("="*50)
    print(f"\nInput shapes:")
    print(f"  - input.1 (features): {features.shape} (dtype: float32)")
    print(f"  - indices_input: {indices.shape} (dtype: int32)")
    print(f"\nOutput files:")
    print(f"  - {features_tar_path}")
    print(f"  - {indices_tar_path}")
    print(f"\nCalibration size: {saved_count}")


def prepare_from_existing_points(
    points_dir: str,
    output_dir: str,
    num_samples: int = 32
):
    """
    Prepare the calibration dataset from existing point cloud bin files
    
    Args:
        points_dir: Directory containing point cloud bin files
        output_dir: Output directory
        num_samples: Number of samples
    """
    points_dir = Path(points_dir)
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    indices_dir = output_dir / "indices"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all bin files
    bin_files = sorted(points_dir.glob("*.bin"))
    
    if len(bin_files) == 0:
        raise FileNotFoundError(f"No .bin files found in {points_dir}")
    
    print(f"Found {len(bin_files)} point cloud files")
    
    # Sampling
    sample_indices = np.linspace(0, len(bin_files) - 1, min(num_samples, len(bin_files)), dtype=int)
    
    for i, idx in enumerate(sample_indices):
        bin_file = bin_files[idx]
        print(f"Processing {i+1}/{len(sample_indices)}: {bin_file.name}")
        
        # Load point cloud
        points = load_points_from_bin(str(bin_file))
        
        # Preprocess
        features, indices = preprocess_point_cloud(points)
        
        # Save
        np.save(features_dir / f"sample_{i:04d}.npy", features)
        np.save(indices_dir / f"sample_{i:04d}.npy", indices)
    
    # Pack into tar.gz
    print("\nCreating tar.gz archives...")
    
    features_tar_path = output_dir / "features.tar.gz"
    with tarfile.open(features_tar_path, "w:gz") as tar:
        for npy_file in sorted(features_dir.glob("*.npy")):
            tar.add(npy_file, arcname=npy_file.name)
    
    indices_tar_path = output_dir / "indices.tar.gz"
    with tarfile.open(indices_tar_path, "w:gz") as tar:
        for npy_file in sorted(indices_dir.glob("*.npy")):
            tar.add(npy_file, arcname=npy_file.name)
    
    print(f"\nCreated:")
    print(f"  - {features_tar_path}")
    print(f"  - {indices_tar_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare calibration dataset for CenterPoint NPU quantization")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="CenterPoint data root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./calib_data",
        help="Output directory for calibration data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of samples for calibration"
    )
    parser.add_argument(
        "--from_points_dir",
        type=str,
        default=None,
        help="If specified, prepare from existing points bin files instead of raw dataset"
    )
    
    args = parser.parse_args()
    
    if args.from_points_dir:
        prepare_from_existing_points(
            points_dir=args.from_points_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    else:
        prepare_calibration_dataset(
            data_root=args.data_root,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
