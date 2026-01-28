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
#include <string>
#include <array>

namespace centerpoint {

// Class names for nuScenes dataset
const std::vector<std::string> CLASS_NAMES = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
};

// Task configuration: class indices for each task
// Task 0: car (1 class)
// Task 1: truck, construction_vehicle (2 classes)
// Task 2: bus, trailer (2 classes)
// Task 3: barrier (1 class)
// Task 4: motorcycle, bicycle (2 classes)
// Task 5: pedestrian, traffic_cone (2 classes)
const std::vector<int> TASK_CLASS_OFFSETS = {0, 1, 3, 5, 6, 8};
const std::vector<int> TASK_NUM_CLASSES = {1, 2, 2, 1, 2, 2};
const int NUM_TASKS = 6;
const int OUTPUTS_PER_TASK = 7;  // reg, height, dim, rot, vel, score, cls

// Configuration structure
struct Config {
    // Voxel generator config
    std::array<float, 3> voxel_size = {0.2f, 0.2f, 8.0f};
    std::array<float, 6> pc_range = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
    int max_points_per_voxel = 20;
    int max_voxels = 30000;
    int num_input_features = 5;  // x, y, z, intensity, time_lag
    
    // Grid size (computed from voxel_size and pc_range)
    int grid_x = 512;  // (51.2 - (-51.2)) / 0.2 = 512
    int grid_y = 512;
    int grid_z = 1;
    
    // Test config
    std::array<float, 6> post_center_limit_range = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
    float score_threshold = 0.1f;
    float nms_iou_threshold = 0.2f;
    int max_per_sample = 500;
    int out_size_factor = 4;
    
    // Output BEV size (128x128 for out_size_factor=4)
    int bev_h = 128;
    int bev_w = 128;
    
    // Model input config
    int max_pillars = 30000;
    int max_points_per_pillar = 20;
    int num_pillar_features = 10;  // x, y, z, intensity, time_lag, x_c, y_c, z_c, x_p, y_p
};

// 3D Detection result
struct Detection3D {
    float x, y, z;           // Center position
    float w, l, h;           // Dimensions (width, length, height)
    float yaw;               // Rotation angle (radians)
    float vx, vy;            // Velocity
    float score;             // Confidence score
    int label;               // Class label
};

// Detection result container
struct DetectionResult {
    std::vector<Detection3D> detections;
    int num_detections;
};

// Voxel data structure
struct VoxelData {
    std::vector<float> voxels;       // [num_voxels, max_points, 5]
    std::vector<int> coors;          // [num_voxels, 3] (z, y, x)
    std::vector<int> num_points;     // [num_voxels]
    int num_voxels;
};

// Pillar input data
struct PillarInput {
    std::vector<float> features;     // [1, 10, max_pillars, max_points]
    std::vector<int> indices;        // [1, max_pillars, 2]
    int num_pillars;
};

} // namespace centerpoint
