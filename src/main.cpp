/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2025, AXERA Semiconductor Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */
/*
 * Author: GUOFANGMING
 */ 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "centerpoint_common.hpp"
#include "data_loader.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "visualization.hpp"
#include "utils.hpp"
#include "timer.hpp"

namespace middleware {
    void prepare_io(AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T* io_data) {
        memset(io_data, 0, sizeof(AX_ENGINE_IO_T));
        
        io_data->nInputSize = info->nInputSize;
        io_data->nOutputSize = info->nOutputSize;
        io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
        io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
        
        memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);
        memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
        
        for (uint32_t i = 0; i < info->nInputSize; ++i) {
            auto& meta = info->pInputs[i];
            auto& buffer = io_data->pInputs[i];
            buffer.nSize = meta.nSize;
            int ret = AX_SYS_MemAlloc((AX_U64*)(&buffer.phyAddr), &buffer.pVirAddr, 
                                       meta.nSize, 128, (const AX_S8*)"centerpoint");
            if (ret != 0) {
                std::cerr << "Failed to allocate input buffer " << i << std::endl;
            }
        }
        
        for (uint32_t i = 0; i < info->nOutputSize; ++i) {
            auto& meta = info->pOutputs[i];
            auto& buffer = io_data->pOutputs[i];
            buffer.nSize = meta.nSize;
            int ret = AX_SYS_MemAlloc((AX_U64*)(&buffer.phyAddr), &buffer.pVirAddr,
                                       meta.nSize, 128, (const AX_S8*)"centerpoint");
            if (ret != 0) {
                std::cerr << "Failed to allocate output buffer " << i << std::endl;
            }
        }
    }
    
    void free_io(AX_ENGINE_IO_T* io) {
        if (!io) return;
        
        for (uint32_t i = 0; i < io->nInputSize; ++i) {
            if (io->pInputs[i].phyAddr != 0) {
                AX_SYS_MemFree(io->pInputs[i].phyAddr, io->pInputs[i].pVirAddr);
            }
        }
        for (uint32_t i = 0; i < io->nOutputSize; ++i) {
            if (io->pOutputs[i].phyAddr != 0) {
                AX_SYS_MemFree(io->pOutputs[i].phyAddr, io->pOutputs[i].pVirAddr);
            }
        }
        delete[] io->pInputs;
        delete[] io->pOutputs;
    }
}

struct Args {
    std::string model_path;
    std::string config_json;
    std::string data_dir;
    std::string output_dir = "./inference_results";
    float score_thr = 0.1f;
    int fps = 10;
    int num_samples = -1;  // -1 means all
    bool visualize = true;
};

bool parse_args(int argc, char* argv[], Args& args) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.axmodel> <config_json> <data_dir> [options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  <model.axmodel>        Path to AX model file (*.axmodel)" << std::endl;
        std::cerr << "  <config_json>          Path to model config JSON file" << std::endl;
        std::cerr << "  <data_dir>             Path to extracted data directory" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --output-dir <dir>     Output directory (default: ./inference_results)" << std::endl;
        std::cerr << "  --score-thr <float>    Score threshold (default: 0.1)" << std::endl;
        std::cerr << "  --fps <int>            Video FPS (default: 10)" << std::endl;
        std::cerr << "  --num-samples <int>    Number of samples to process (default: all)" << std::endl;
        std::cerr << "  --no-visualize         Disable visualization" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " centerpoint.axmodel config.json ./extracted_data --output-dir ./results" << std::endl;
        return false;
    }
    
    args.model_path = argv[1];
    args.config_json = argv[2];
    args.data_dir = argv[3];
    
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--output-dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--score-thr" && i + 1 < argc) {
            args.score_thr = std::stof(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            args.fps = std::stoi(argv[++i]);
        } else if (arg == "--num-samples" && i + 1 < argc) {
            args.num_samples = std::stoi(argv[++i]);
        } else if (arg == "--no-visualize") {
            args.visualize = false;
        }
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return -1;
    }
    
    // Load configuration
    centerpoint::Config config;
    if (!centerpoint::load_config(args.config_json, config)) {
        std::cerr << "Failed to load config from: " << args.config_json << std::endl;
        return -1;
    }
    
    // Load sample index
    std::vector<centerpoint::SampleInfo> samples;
    if (!centerpoint::load_sample_index(args.data_dir, samples)) {
        std::cerr << "Failed to load sample index from: " << args.data_dir << std::endl;
        return -1;
    }
    
    // Limit number of samples if specified
    if (args.num_samples > 0 && args.num_samples < static_cast<int>(samples.size())) {
        samples.resize(args.num_samples);
    }
    
    
    // Create output directories
    system(("mkdir -p " + args.output_dir).c_str());
    std::string images_dir = centerpoint::join_path(args.output_dir, "images");
    if (args.visualize) {
        system(("mkdir -p " + images_dir).c_str());
    }
    
    // Initialize AX Engine
    AX_SYS_Init();
    
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    int ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_Init failed: " << ret << std::endl;
        return -1;
    }
    
    // Load model
    std::ifstream model_file(args.model_path, std::ios::binary);
    if (!model_file) {
        std::cerr << "Failed to open model file: " << args.model_path << std::endl;
        return -1;
    }
    
    model_file.seekg(0, std::ios::end);
    size_t model_size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);
    
    std::vector<char> model_buffer(model_size);
    model_file.read(model_buffer.data(), model_size);
    model_file.close();
    
    std::cout << "[Model] " << args.model_path << " (" << model_size / 1024 / 1024 << " MB)" << std::endl;
    
    // Create handle
    AX_ENGINE_HANDLE handle;
    ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
    if (ret != 0) {
        std::cerr << "AX_ENGINE_CreateHandle failed: " << ret << std::endl;
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }
    
    // Create context
    ret = AX_ENGINE_CreateContext(handle);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_CreateContext failed: " << ret << std::endl;
        AX_ENGINE_DestroyHandle(handle);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }
    
    // Get IO info
    AX_ENGINE_IO_INFO_T* io_info;
    ret = AX_ENGINE_GetIOInfo(handle, &io_info);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_GetIOInfo failed: " << ret << std::endl;
        AX_ENGINE_DestroyHandle(handle);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }
    
    // Print IO info (compact)
    std::cout << "Model: " << io_info->nInputSize << " inputs, " 
              << io_info->nOutputSize << " outputs" << std::endl;
    
    // Allocate IO
    AX_ENGINE_IO_T io_data;
    middleware::prepare_io(io_info, &io_data);
    
    // Find input indices - check indices first to avoid "indices_input" matching "input"
    int features_input_idx = -1;
    int indices_input_idx = -1;
    for (uint32_t i = 0; i < io_info->nInputSize; ++i) {
        std::string name = io_info->pInputs[i].pName;
        if (name.find("indices") != std::string::npos) {
            indices_input_idx = i;
        } else if (name.find("input") != std::string::npos || name.find("feature") != std::string::npos) {
            features_input_idx = i;
        }
    }
    
    // Default to first two inputs if not found by name
    if (features_input_idx < 0) features_input_idx = 0;
    if (indices_input_idx < 0) indices_input_idx = 1;
    
    // Timing statistics
    std::vector<float> load_times;
    std::vector<float> preprocess_times;
    std::vector<float> inference_times;
    std::vector<float> postprocess_times;
    std::vector<float> visualize_times;
    std::vector<float> total_times;
    
    centerpoint::Timer scene_timer;
    scene_timer.start();
    
    int total_detections = 0;
    
    // Process each sample
    for (size_t idx = 0; idx < samples.size(); ++idx) {
        const auto& sample = samples[idx];
        centerpoint::Timer frame_timer;
        centerpoint::Timer step_timer;
        
        // Load point cloud
        step_timer.start();
        std::vector<float> points;
        int num_points;
        std::string points_path = centerpoint::join_path(args.data_dir, sample.points_path);
        if (!centerpoint::load_points(points_path, points, num_points)) {
            std::cerr << "Failed to load points: " << points_path << std::endl;
            continue;
        }
        float load_time = step_timer.cost_ms();
        load_times.push_back(load_time);
        
        // Preprocess: voxelization and pillar feature extraction
        step_timer.reset();
        std::vector<float> features;
        std::vector<int> indices;
        centerpoint::preprocess(points, num_points, config, features, indices);
        float preprocess_time = step_timer.cost_ms();
        preprocess_times.push_back(preprocess_time);
        
        // Copy inputs to engine buffers
        size_t features_size = features.size() * sizeof(float);
        size_t indices_size = indices.size() * sizeof(int);
        
        memcpy(io_data.pInputs[features_input_idx].pVirAddr, features.data(),
               std::min(features_size, (size_t)io_data.pInputs[features_input_idx].nSize));
        memcpy(io_data.pInputs[indices_input_idx].pVirAddr, indices.data(),
               std::min(indices_size, (size_t)io_data.pInputs[indices_input_idx].nSize));
        
        // Run inference
        step_timer.reset();
        ret = AX_ENGINE_RunSync(handle, &io_data);
        float inference_time = step_timer.cost_ms();
        inference_times.push_back(inference_time);
        
        if (ret != 0) {
            std::cerr << "AX_ENGINE_RunSync failed: " << ret << std::endl;
            continue;
        }
        
        // Get outputs
        step_timer.reset();
        std::vector<float*> outputs;
        std::vector<size_t> output_sizes;
        for (uint32_t i = 0; i < io_info->nOutputSize; ++i) {
            outputs.push_back(reinterpret_cast<float*>(io_data.pOutputs[i].pVirAddr));
            output_sizes.push_back(io_data.pOutputs[i].nSize / sizeof(float));
        }
        
        // Post-process
        centerpoint::DetectionResult result = centerpoint::postprocess(
            outputs, output_sizes, config, args.score_thr);
        float postprocess_time = step_timer.cost_ms();
        postprocess_times.push_back(postprocess_time);
        
        total_detections += result.num_detections;
        
        // Visualize
        float visualize_time = 0.0f;
        if (args.visualize) {
            step_timer.reset();
            cv::Mat vis_img = centerpoint::visualize_bev(
                points, num_points, result, config, idx, 35.0f, args.score_thr);
            
            char frame_filename[64];
            snprintf(frame_filename, sizeof(frame_filename), "frame_%06zu.png", idx);
            std::string output_path = centerpoint::join_path(images_dir, frame_filename);
            cv::imwrite(output_path, vis_img);
            visualize_time = step_timer.cost_ms();
        }
        visualize_times.push_back(visualize_time);
        
        float total_time = frame_timer.cost_ms();
        total_times.push_back(total_time);
        
        // Show progress
        float elapsed_ms = scene_timer.cost_ms();
        centerpoint::print_progress_bar(idx + 1, samples.size(), "Processing:", elapsed_ms);
    }
    
    // Print statistics (compact)
    if (!inference_times.empty()) {
        float avg_inference = std::accumulate(inference_times.begin(), inference_times.end(), 0.0f) / inference_times.size();
        float avg_total = std::accumulate(total_times.begin(), total_times.end(), 0.0f) / total_times.size();
        
        std::cout << "\n[Performance] " << samples.size() << " samples"
                  << ", Inference: " << avg_inference << "ms"
                  << ", Total: " << avg_total << "ms"
                  << ", FPS: " << 1000.0f / avg_total << std::endl;
        std::cout << "[Detections] " << total_detections << " total" << std::endl;
    }
    
    // Create video
    if (args.visualize) {
        std::string video_path = centerpoint::join_path(args.output_dir, "centerpoint_detection.mp4");
        std::string actual_video_path;
        centerpoint::create_video_from_images(images_dir, video_path, actual_video_path, args.fps);
    }
    
    std::cout << "[Done] Results saved to: " << args.output_dir << std::endl;
    
    // Cleanup
    middleware::free_io(&io_data);
    AX_ENGINE_DestroyHandle(handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    
    return 0;
}
