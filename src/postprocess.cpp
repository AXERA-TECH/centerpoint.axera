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
#include "postprocess.hpp"
#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace centerpoint {

void decode_task_output(const TaskOutput& output,
                        int task_idx,
                        const Config& config,
                        std::vector<Detection3D>& detections) {
    int H = config.bev_h;
    int W = config.bev_w;
    float voxel_size_x = config.voxel_size[0];
    float voxel_size_y = config.voxel_size[1];
    float pc_range_x = config.pc_range[0];
    float pc_range_y = config.pc_range[1];
    int out_size_factor = config.out_size_factor;
    float score_threshold = config.score_threshold;
    int class_offset = TASK_CLASS_OFFSETS[task_idx];
    
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = h * W + w;
            
            float score = output.score[idx];
            if (score < score_threshold) continue;
            
            Detection3D det;
            
            // Decode center position
            float reg_x = output.reg[0 * H * W + idx];
            float reg_y = output.reg[1 * H * W + idx];
            det.x = (w + reg_x) * out_size_factor * voxel_size_x + pc_range_x;
            det.y = (h + reg_y) * out_size_factor * voxel_size_y + pc_range_y;
            det.z = output.height[idx];
            
            // Decode dimensions
            det.l = output.dim[0 * H * W + idx];  // length
            det.h = output.dim[1 * H * W + idx];  // height
            det.w = output.dim[2 * H * W + idx];  // width
            
            // Decode rotation
            float sin_val = output.rot[0 * H * W + idx];
            float cos_val = output.rot[1 * H * W + idx];
            det.yaw = std::atan2(sin_val, cos_val);
            
            // Decode velocity
            det.vx = output.vel[0 * H * W + idx];
            det.vy = output.vel[1 * H * W + idx];
            
            det.score = score;
            det.label = output.cls[idx] + class_offset;
            
            detections.push_back(det);
        }
    }
}

void nms_bev(std::vector<Detection3D>& detections,
             float nms_threshold,
             std::vector<int>& keep_indices) {
    int n = detections.size();
    if (n == 0) return;
    
    // Sort by score descending
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&detections](int a, int b) {
        return detections[a].score > detections[b].score;
    });
    
    // Compute box corners (aligned BEV boxes)
    std::vector<float> x1(n), y1(n), x2(n), y2(n), areas(n);
    for (int i = 0; i < n; ++i) {
        const auto& det = detections[i];
        x1[i] = det.x - det.l / 2.0f;
        y1[i] = det.y - det.w / 2.0f;
        x2[i] = det.x + det.l / 2.0f;
        y2[i] = det.y + det.w / 2.0f;
        areas[i] = det.w * det.l;
    }
    
    std::vector<bool> suppressed(n, false);
    
    for (int _i = 0; _i < n; ++_i) {
        int i = order[_i];
        if (suppressed[i]) continue;
        
        keep_indices.push_back(i);
        
        for (int _j = _i + 1; _j < n; ++_j) {
            int j = order[_j];
            if (suppressed[j]) continue;
            
            // Compute intersection
            float ix1 = std::max(x1[i], x1[j]);
            float iy1 = std::max(y1[i], y1[j]);
            float ix2 = std::min(x2[i], x2[j]);
            float iy2 = std::min(y2[i], y2[j]);
            
            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            
            // Compute IoU
            float union_area = areas[i] + areas[j] - inter;
            float iou = inter / std::max(union_area, 1e-6f);
            
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
}

DetectionResult postprocess(const std::vector<float*>& outputs,
                            const std::vector<size_t>& output_sizes,
                            const Config& config,
                            float score_thr) {
    DetectionResult result;
    int H = config.bev_h;
    int W = config.bev_w;
    
    // Check if we have the expected number of outputs
    int expected_outputs = NUM_TASKS * OUTPUTS_PER_TASK;  // 42
    int num_outputs = static_cast<int>(outputs.size());
    
    if (num_outputs < expected_outputs) {
        // If fewer outputs, try to process what we have
        // Some models might have different output configurations
        std::cerr << "Warning: Expected " << expected_outputs 
                  << " outputs but got " << num_outputs << std::endl;
    }
    
    // Determine number of tasks based on available outputs
    int num_tasks = std::min(NUM_TASKS, num_outputs / OUTPUTS_PER_TASK);
    
    // Process each task
    // Model has 42 outputs: 7 outputs per task, 6 tasks
    // Order per task: reg, height, dim, rot, vel, score, cls
    for (int task_idx = 0; task_idx < num_tasks; ++task_idx) {
        int base_idx = task_idx * OUTPUTS_PER_TASK;
        
        // Validate we have enough outputs for this task
        if (base_idx + 6 >= num_outputs) break;
        
        TaskOutput task_output;
        
        // reg: [1, 2, H, W]
        float* reg_ptr = outputs[base_idx + 0];
        task_output.reg.assign(reg_ptr, reg_ptr + 2 * H * W);
        
        // height: [1, 1, H, W]
        float* height_ptr = outputs[base_idx + 1];
        task_output.height.assign(height_ptr, height_ptr + H * W);
        
        // dim: [1, 3, H, W]
        float* dim_ptr = outputs[base_idx + 2];
        task_output.dim.assign(dim_ptr, dim_ptr + 3 * H * W);
        
        // rot: [1, 2, H, W]
        float* rot_ptr = outputs[base_idx + 3];
        task_output.rot.assign(rot_ptr, rot_ptr + 2 * H * W);
        
        // vel: [1, 2, H, W]
        float* vel_ptr = outputs[base_idx + 4];
        task_output.vel.assign(vel_ptr, vel_ptr + 2 * H * W);
        
        // score: [1, H, W] (already sigmoid applied)
        float* score_ptr = outputs[base_idx + 5];
        task_output.score.assign(score_ptr, score_ptr + H * W);
        
        // cls: [1, H, W] (already argmax applied, stored as float)
        float* cls_ptr = outputs[base_idx + 6];
        task_output.cls.resize(H * W);
        for (int i = 0; i < H * W; ++i) {
            task_output.cls[i] = static_cast<int>(cls_ptr[i]);
        }
        
        // Decode detections for this task
        decode_task_output(task_output, task_idx, config, result.detections);
    }
    
    if (result.detections.empty()) {
        result.num_detections = 0;
        return result;
    }
    
    // Apply NMS
    std::vector<int> keep_indices;
    nms_bev(result.detections, config.nms_iou_threshold, keep_indices);
    
    // Filter detections
    std::vector<Detection3D> filtered;
    for (int idx : keep_indices) {
        if (result.detections[idx].score >= score_thr) {
            filtered.push_back(result.detections[idx]);
        }
    }
    
    // Limit to max_per_sample
    if (static_cast<int>(filtered.size()) > config.max_per_sample) {
        // Sort by score and keep top N
        std::sort(filtered.begin(), filtered.end(), 
                  [](const Detection3D& a, const Detection3D& b) {
                      return a.score > b.score;
                  });
        filtered.resize(config.max_per_sample);
    }
    
    result.detections = std::move(filtered);
    result.num_detections = result.detections.size();
    
    return result;
}

} // namespace centerpoint
