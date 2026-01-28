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
#pragma once

#include <vector>
#include "centerpoint_common.hpp"

namespace centerpoint {

// Model output structure (per task)
struct TaskOutput {
    std::vector<float> reg;      // [2, H, W] registration offset
    std::vector<float> height;   // [1, H, W] height
    std::vector<float> dim;      // [3, H, W] dimensions (l, h, w)
    std::vector<float> rot;      // [2, H, W] rotation (sin, cos)
    std::vector<float> vel;      // [2, H, W] velocity
    std::vector<float> score;    // [H, W] confidence (after sigmoid)
    std::vector<int> cls;        // [H, W] class index (after argmax)
};

// Decode single task output to detections
void decode_task_output(const TaskOutput& output,
                        int task_idx,
                        const Config& config,
                        std::vector<Detection3D>& detections);

// BEV NMS (aligned bounding box NMS)
void nms_bev(std::vector<Detection3D>& detections,
             float nms_threshold,
             std::vector<int>& keep_indices);

// Post-process all model outputs
DetectionResult postprocess(const std::vector<float*>& outputs,
                            const std::vector<size_t>& output_sizes,
                            const Config& config,
                            float score_thr = 0.1f);

} // namespace centerpoint
