[English](./README_EN.md) | [简体中文](./README.md)

# CenterPoint Inference

CenterPoint 3D Object Detection DEMO on Axera NPU

## Supported Platform

- [x] AX650

## Model and Data Download

The pre-converted AXModel, inference data, and configuration files can be downloaded from Hugging Face:

🤗 **[AXERA-TECH/centerpoint](https://huggingface.co/AXERA-TECH/centerpoint)**

Download contents include:
- `centerpoint.axmodel` - Pre-converted AX650 NPU model (w8a16 quantization, Pulsar2 4.2 compatible)
- `extracted_data/` - Inference test data
  - `config.json` - Model configuration
  - `sample_index.json` - Sample index
  - `points/` - Point cloud data

```bash
# Download using Git LFS
git lfs install
git clone https://huggingface.co/AXERA-TECH/centerpoint

# Or use huggingface-cli
pip install huggingface_hub
huggingface-cli download AXERA-TECH/centerpoint --local-dir ./centerpoint_hf
```

## Project Structure

```
centerpoint.axera/
├── CMakeLists.txt          # Build configuration
├── build650.sh             # Build script for AX650
├── README.md               # Chinese documentation
├── README_EN.md            # This file (English)
├── toolchains/
│   └── aarch64-none-linux-gnu.toolchain.cmake  # Cross-compilation toolchain
├── include/                # Header files
│   ├── centerpoint_common.hpp
│   ├── data_loader.hpp
│   ├── preprocess.hpp
│   ├── postprocess.hpp
│   ├── visualization.hpp
│   ├── utils.hpp
│   └── timer.hpp
├── src/                    # Source files
│   ├── main.cpp
│   ├── centerpoint_common.cpp
│   ├── data_loader.cpp
│   ├── preprocess.cpp
│   ├── postprocess.cpp
│   ├── visualization.cpp
│   └── utils.cpp
├── onnx_and_ax_demo/       # Python inference scripts
│   ├── inference_axmodel.py    # AXEngine Python inference
│   ├── inference_onnx.py       # ONNX inference
│   ├── extract_data_simple.py  # Data extraction script
│   └── prepare_calib_data.py   # Calibration data preparation
└── centerpoint_export/     # ONNX export related
```

## Dependencies

- OpenCV (>= 3.0)
- AXERA BSP (msp/out directory) - AX650 specific
- CMake (>= 3.13)
- C++14 compiler
- Cross-compilation toolchain (for x86_64 hosts targeting aarch64)

## Building

### Automated Build (Recommended)

The project provides an automated build script for AX650:

```bash
./build650.sh
```

The build script will automatically:
1. Check and verify system dependencies (cmake, wget, unzip, tar, git, make)
2. Download and setup OpenCV library for aarch64
3. Clone and setup BSP SDK for AX650
4. Download and setup cross-compilation toolchain (for x86_64 hosts)
5. Configure CMake and build the project

**Note**:
- On first run, the script will download ~500MB of dependencies. Subsequent runs will reuse cached files.
- Build outputs are stored in `build_ax650/` directory

### Manual Build

If you prefer to build manually:

```bash
mkdir build_ax650 && cd build_ax650
cmake -DBSP_MSP_DIR=/path/to/ax650/msp/out -DAXERA_TARGET_CHIP=ax650 ..
make -j$(nproc)
```

#### Manual Dependency Setup

1. **OpenCV**: Download from [here](https://github.com/AXERA-TECH/ax-samples/releases/download/v0.1/opencv-aarch64-linux-gnu-gcc-7.5.0.zip) and extract to `3rdparty/`
2. **BSP SDK**: Clone from `https://github.com/AXERA-TECH/ax650n_bsp_sdk.git`
3. **Toolchain**: Download from [ARM](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz) and extract

## Usage

```bash
./centerpoint_inference <model.axmodel> <config_json> <data_dir> [options]
```

### Arguments

- `model.axmodel`: Path to CenterPoint AXModel file
- `config_json`: Path to configuration JSON file
- `data_dir`: Path to extracted data directory (should contain sample_index.json)

### Options

- `--output-dir <dir>`: Output directory (default: ./inference_results)
- `--score-thr <float>`: Score threshold (default: 0.1)
- `--fps <int>`: Video FPS (default: 10)
- `--num-samples <int>`: Number of samples to process (default: all)
- `--no-visualize`: Disable visualization

### Example

```bash
# Using extracted data from onnx_and_ax_demo directory
./centerpoint_inference \
    centerpoint.axmodel \
    ./onnx_and_ax_demo/extracted_data/config.json \
    ./onnx_and_ax_demo/extracted_data \
    --output-dir ./results \
    --score-thr 0.5 \
    --fps 10
```

## Data Preparation

### Using Extraction Script

Use `onnx_and_ax_demo/extract_data_simple.py` to extract point cloud data from nuScenes dataset:

```bash
python onnx_and_ax_demo/extract_data_simple.py \
    --data-root /path/to/nuscenes/data \
    --output-dir ./extracted_data \
    --num-samples 50
```

### Data Directory Structure

```
extracted_data/
├── config.json           # Model configuration
├── sample_index.json     # Sample index
├── points/               # Point cloud data
│   ├── 000000.bin
│   ├── 000001.bin
│   └── ...
└── gt_annotations/       # Ground truth annotations (optional)
    ├── 000000.json
    └── ...
```

## Python Inference

The project also includes Python reference implementations:

### AXEngine Inference

```bash
python onnx_and_ax_demo/inference_axmodel.py \
    centerpoint.axmodel \
    config.json \
    ./extracted_data \
    --output-dir ./results \
    --visualize
```

### ONNX Inference

```bash
python onnx_and_ax_demo/inference_onnx.py \
    centerpoint.onnx \
    config.json \
    ./extracted_data \
    --output-dir ./results
```

## Model Conversion

### ONNX Export

Refer to the scripts in `centerpoint_export` directory to export ONNX model.

### ONNX to AXModel Conversion

Use Pulsar2 tool to convert ONNX model to AXModel. For detailed conversion guidance, please refer to [Pulsar2 Documentation](https://pulsar2-docs.readthedocs.io/en/latest/index.html).

## Output

### Output Structure

```
output_dir/
├── images/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
└── centerpoint_detection.mp4
```

### Visualization Description

Each frame visualization includes:
- **Point Cloud**: Colored point cloud (color indicates distance)
- **Detection Boxes**: 3D bounding boxes with different colors for different classes
- **Class Legend**: Shows color identification for each class

### Supported Classes

| Class ID | Class Name | Color |
|----------|------------|-------|
| 0 | car | Blue |
| 1 | truck | Orange |
| 2 | construction_vehicle | Red |
| 3 | bus | Yellow |
| 4 | trailer | Purple |
| 5 | barrier | Cyan |
| 6 | motorcycle | Red |
| 7 | bicycle | Green |
| 8 | pedestrian | Magenta |
| 9 | traffic_cone | Yellow |

## Running Example

Run on AX650:

```bash
./centerpoint_inference centerpoint.axmodel ./extracted_data/config.json extracted_data/ --output-dir ./results --score-thr 0.5 --fps 10
```

Output:

```
[Config] BEV: 128x128, voxels: 60000, score_thr: 0.1
[Data] 50 samples loaded
[Model] centerpoint.axmodel (71 MB)
Model: 2 inputs, 42 outputs
Processing: [========================================] 100% [50/50] 259.7fps, ETA: 00:00

[Performance] 50 samples, Inference: 88.5788ms, Total: 177.44ms, FPS: 5.63572
[Detections] 1029 total
Video: 50 frames -> ./results/centerpoint_detection_video.avi
[Done] Results saved to: ./results
```

## Visualization Result

![CenterPoint Detection Result](./asset/output.gif)

## Performance

Typical performance on AX650:

| Stage | Time |
|-------|------|
| NPU Inference | ~88 ms |
| Total | ~177 ms |
| FPS | ~5.6 FPS |

## Technical Discussion

- Github issues
- QQ Group: 139953715

