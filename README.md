[English](./README_EN.md) | [简体中文](./README.md)

# CenterPoint 推理

CenterPoint 3D 目标检测 DEMO on Axera NPU

## 支持平台

- [x] AX650

## 模型和数据下载

已转换的 AXModel 模型、推理数据和配置文件可以从 Hugging Face 下载：

🤗 **[AXERA-TECH/centerpoint](https://huggingface.co/AXERA-TECH/centerpoint)**

下载内容包括：
- `centerpoint.axmodel` - 已转换的 AX650 NPU 模型（w8a16 量化，Pulsar2 4.2 兼容）
- `extracted_data/` - 推理测试数据
  - `config.json` - 模型配置
  - `sample_index.json` - 样本索引
  - `points/` - 点云数据

```bash
# 使用 Git LFS 下载
git lfs install
git clone https://huggingface.co/AXERA-TECH/centerpoint

# 或使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download AXERA-TECH/centerpoint --local-dir ./centerpoint_hf
```

## 项目结构

```
centerpoint.axera/
├── CMakeLists.txt          # 构建配置文件
├── build650.sh             # AX650 构建脚本
├── README.md               # 本文档（中文版）
├── README_EN.md            # 英文版文档
├── toolchains/
│   └── aarch64-none-linux-gnu.toolchain.cmake  # 交叉编译工具链
├── include/                # 头文件
│   ├── centerpoint_common.hpp
│   ├── data_loader.hpp
│   ├── preprocess.hpp
│   ├── postprocess.hpp
│   ├── visualization.hpp
│   ├── utils.hpp
│   └── timer.hpp
├── src/                    # 源文件
│   ├── main.cpp
│   ├── centerpoint_common.cpp
│   ├── data_loader.cpp
│   ├── preprocess.cpp
│   ├── postprocess.cpp
│   ├── visualization.cpp
│   └── utils.cpp
├── onnx_and_ax_demo/       # Python 推理脚本
│   ├── inference_axmodel.py    # AXEngine Python 推理
│   ├── inference_onnx.py       # ONNX 推理
│   ├── extract_data_simple.py  # 数据提取脚本
│   └── prepare_calib_data.py   # 校准数据准备
└── centerpoint_export/     # ONNX 导出相关
```

## 依赖项

- OpenCV (>= 3.0)
- AXERA BSP (msp/out 目录) - AX650 专用
- CMake (>= 3.13)
- C++14 编译器
- 交叉编译工具链（用于在 x86_64 主机上编译 aarch64 目标）

## 构建

### 自动化构建（推荐）

项目提供了 AX650 的自动化构建脚本：

```bash
./build650.sh
```

构建脚本将自动：
1. 检查并验证系统依赖项（cmake、wget、unzip、tar、git、make）
2. 下载并设置适用于 aarch64 的 OpenCV 库
3. 克隆并设置 AX650 的 BSP SDK
4. 下载并设置交叉编译工具链（适用于 x86_64 主机）
5. 使用 CMake 配置并构建项目

**注意**：
- 首次运行时，脚本将下载约 500MB 的依赖项。后续运行将重用缓存文件。
- 构建输出存储在 `build_ax650/` 目录中

### 手动构建

如果您更喜欢手动构建：

```bash
mkdir build_ax650 && cd build_ax650
cmake -DBSP_MSP_DIR=/path/to/ax650/msp/out -DAXERA_TARGET_CHIP=ax650 ..
make -j$(nproc)
```

#### 手动依赖项设置

1. **OpenCV**：从[这里](https://github.com/AXERA-TECH/ax-samples/releases/download/v0.1/opencv-aarch64-linux-gnu-gcc-7.5.0.zip)下载并解压到 `3rdparty/`
2. **BSP SDK**：从 `https://github.com/AXERA-TECH/ax650n_bsp_sdk.git` 克隆
3. **工具链**：从 [ARM](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz) 下载并解压

## 使用方法

```bash
./centerpoint_inference <model.axmodel> <config_json> <data_dir> [options]
```

### 参数

- `model.axmodel`：CenterPoint AXModel 文件路径
- `config_json`：配置文件 JSON 路径
- `data_dir`：提取的数据目录路径（应包含 sample_index.json）

### 选项

- `--output-dir <dir>`：输出目录（默认：./inference_results）
- `--score-thr <float>`：分数阈值（默认：0.1）
- `--fps <int>`：视频帧率（默认：10）
- `--num-samples <int>`：处理的样本数量（默认：全部）
- `--no-visualize`：禁用可视化

### 示例

```bash
# 使用 onnx_and_ax_demo 目录中的提取数据
./centerpoint_inference \
    centerpoint.axmodel \
    ./onnx_and_ax_demo/extracted_data/config.json \
    ./onnx_and_ax_demo/extracted_data \
    --output-dir ./results \
    --score-thr 0.5 \
    --fps 10
```

## 数据准备

### 使用提取脚本

使用 `onnx_and_ax_demo/extract_data_simple.py` 从 nuScenes 数据集提取点云数据：

```bash
python onnx_and_ax_demo/extract_data_simple.py \
    --data-root /path/to/nuscenes/data \
    --output-dir ./extracted_data \
    --num-samples 50
```

### 数据目录结构

```
extracted_data/
├── config.json           # 模型配置
├── sample_index.json     # 样本索引
├── points/               # 点云数据
│   ├── 000000.bin
│   ├── 000001.bin
│   └── ...
└── gt_annotations/       # 真值标注（可选）
    ├── 000000.json
    └── ...
```

## Python 推理

项目还包含 Python 参考实现：

### AXEngine 推理

```bash
python onnx_and_ax_demo/inference_axmodel.py \
    centerpoint.axmodel \
    config.json \
    ./extracted_data \
    --output-dir ./results \
    --visualize
```

### ONNX 推理

```bash
python onnx_and_ax_demo/inference_onnx.py \
    centerpoint.onnx \
    config.json \
    ./extracted_data \
    --output-dir ./results
```

## 模型转换

### ONNX 导出

参考 `centerpoint_export` 目录中的脚本导出 ONNX 模型。

### ONNX 到 AXModel 转换

使用 Pulsar2 工具将 ONNX 模型转换为 AXModel，详细转换请参考 [Pulsar2 文档](https://pulsar2-docs.readthedocs.io/en/latest/index.html)。

## 输出

### 输出结构

```
output_dir/
├── images/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
└── centerpoint_detection.mp4
```

### 可视化说明

每帧可视化包括：
- **点云**：彩色点云（颜色表示距离）
- **检测框**：不同类别使用不同颜色的 3D 边界框
- **类别图例**：显示各类别的颜色标识

### 支持的类别

| 类别 ID | 类别名称 | 颜色 |
|---------|----------|------|
| 0 | car | 蓝色 |
| 1 | truck | 橙色 |
| 2 | construction_vehicle | 红色 |
| 3 | bus | 黄色 |
| 4 | trailer | 紫色 |
| 5 | barrier | 青色 |
| 6 | motorcycle | 红色 |
| 7 | bicycle | 绿色 |
| 8 | pedestrian | 品红色 |
| 9 | traffic_cone | 黄色 |

## 运行示例

在 AX650 上运行：

```bash
./centerpoint_inference centerpoint.axmodel ./extracted_data/config.json extracted_data/ --output-dir ./results --score-thr 0.5 --fps 10
```

输出：

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

## 可视化结果

![CenterPoint Detection Result](./asset/output.gif)

## 性能

在 AX650 上的典型性能：

| 阶段 | 时间 |
|------|------|
| NPU 推理 | ~88 ms |
| 总耗时 | ~177 ms |
| 帧率 | ~5.6 FPS |

## 技术讨论

- Github issues
- QQ 群: 139953715

