# CenterPoint ONNX & AXEngine Demo

This folder contains standalone inference scripts for CenterPoint that do not depend on the det3d library.

## Overview

The demo consists of three main scripts:

1. **`extract_data.py`** - Extract dataset from det3d format and save to standalone files
2. **`inference_onnx.py`** - Run inference using ONNX Runtime
3. **`inference_axmodel.py`** - Run inference using AXEngine

## Requirements

### For data extraction (only once)
- det3d library and its dependencies
- PyTorch

### For inference (no det3d needed)
```bash
pip install numpy numba onnxruntime tqdm opencv-python
```

For AXEngine inference, you also need [pyaxengine](https://github.com/AXERA-TECH/pyaxengine)


## Usage

### Step 1: Extract Data (requires det3d)

First, extract the dataset to avoid det3d dependency during inference:

```bash
cd path/to/onnx_demo

# Extract 10 samples from validation set
python extract_data.py \
    --config ../configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo.py \
    --output-dir ./extracted_data \
    --num-samples 10 \
    --split val
```

This will create:
```
extracted_data/
├── config.json          # Model and data configuration
├── sample_index.json    # Index of all samples
├── points/              # Point cloud binary files
│   ├── {token}.bin
│   └── ...
└── gt_annotations/      # Ground truth annotations (optional)
    ├── {token}.json
    └── ...
```

### Step 2: Run ONNX Inference (no det3d needed)

```bash
# Make sure you're using the correct Python environment
python inference_onnx.py \
    ../onnx_model/pointpillars.onnx \
    ./extracted_data/config.json \
    ./extracted_data \
    --output-dir ./inference_results \
    --num-samples 10 \
    --visualize \
    --fps 10
```

Options:
- `--score-thr`: Score threshold for filtering detections (default: 0.1)
- `--device`: Device for inference, e.g., 'cuda:0' or 'cpu' (default: cuda:0)
- `--num-samples`: Number of samples to process (default: all)
- `--visualize`: Save BEV visualization images and create video
- `--fps`: Video frames per second (default: 10)

Output:
```
inference_results/
├── images/                              # BEV visualization images
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── centerpoint_detection_onnx.mp4       # Result video
└── results.json                         # Detection results
```

### Step 3: Run AXEngine Inference (no det3d needed)

```bash
python inference_axmodel.py \
    ../centerpoint.axmodel \
    ./extracted_data/config.json \
    ./extracted_data \
    --output-dir ./inference_results_ax \
    --num-samples 10 \
    --visualize \
    --fps 10
```

Output:
```
inference_results_ax/
├── images/                              # BEV visualization images
├── centerpoint_detection_axmodel.mp4    # Result video
└── results.json                         # Detection results
```

## Output Format

Both inference scripts produce results in JSON format:

```json
{
  "token": "sample_000000",
  "boxes": [[x, y, z, w, l, h, theta, vx, vy], ...],
  "scores": [0.95, 0.87, ...],
  "labels": [0, 1, ...],
  "num_detections": 42
}
```

Box format: `[x, y, z, w, l, h, theta, vx, vy]`
- `x, y, z`: Center position in LiDAR coordinates
- `w, l, h`: Width, length, height
- `theta`: Rotation angle (yaw)
- `vx, vy`: Velocity

## Class Labels

| Label | Class Name |
|-------|------------|
| 0 | car |
| 1 | truck |
| 2 | construction_vehicle |
| 3 | bus |
| 4 | trailer |
| 5 | barrier |
| 6 | motorcycle |
| 7 | bicycle |
| 8 | pedestrian |
| 9 | traffic_cone |

## Model Input Format

The PointPillars model expects two inputs:

1. **features**: `[1, 10, max_pillars, max_points_per_pillar]`
   - Channels: x, y, z, intensity, time_lag, x_c, y_c, z_c, x_p, y_p
   
2. **indices**: `[1, max_pillars, 2]`
   - Format: [batch_idx, pillar_idx_in_bev]

Where:
- `max_pillars = 30000`
- `max_points_per_pillar = 20`

## C++ Inference (on AX650 device)

For deployment on AX650, use the C++ inference implementation located in the parent directory:

```bash
# Build the project (on x86_64 host)
cd ..
./build650.sh

# Copy to AX650 device and run
./centerpoint_inference \
    centerpoint.axmodel \
    ./onnx_and_ax_demo/extracted_data/config.json \
    ./onnx_and_ax_demo/extracted_data \
    --output-dir ./results \
    --score-thr 0.3
```

See the main `README.md` in the parent directory for detailed C++ build and usage instructions.

## Configuration

The `config.json` file contains:
- Model parameters (voxel size, point cloud range)
- Test configuration (NMS settings, score thresholds)
- Class names and task definitions
