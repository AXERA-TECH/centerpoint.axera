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
#include "preprocess.hpp"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <algorithm>

namespace centerpoint {

// Hash function for 3D coordinate
struct CoordHash {
    size_t operator()(const std::array<int, 3>& coord) const {
        return std::hash<int>()(coord[0]) ^ 
               (std::hash<int>()(coord[1]) << 10) ^ 
               (std::hash<int>()(coord[2]) << 20);
    }
};

void points_to_voxels(const std::vector<float>& points,
                      int num_points,
                      const Config& config,
                      VoxelData& voxel_data) {
    const float* voxel_size = config.voxel_size.data();
    const float* pc_range = config.pc_range.data();
    int max_points = config.max_points_per_voxel;
    int max_voxels = config.max_voxels;
    
    // Grid size
    int grid_x = config.grid_x;
    int grid_y = config.grid_y;
    int grid_z = config.grid_z;
    
    // Map from coordinate to voxel index
    std::unordered_map<std::array<int, 3>, int, CoordHash> coor_to_voxelidx;
    
    // Allocate output buffers
    int num_features = 5;  // x, y, z, intensity, time_lag
    voxel_data.voxels.resize(max_voxels * max_points * num_features, 0.0f);
    voxel_data.coors.resize(max_voxels * 3, 0);
    voxel_data.num_points.resize(max_voxels, 0);
    voxel_data.num_voxels = 0;
    
    // Process each point
    for (int i = 0; i < num_points; ++i) {
        float x = points[i * 5 + 0];
        float y = points[i * 5 + 1];
        float z = points[i * 5 + 2];
        
        // Check if point is within range
        if (x < pc_range[0] || x >= pc_range[3] ||
            y < pc_range[1] || y >= pc_range[4] ||
            z < pc_range[2] || z >= pc_range[5]) {
            continue;
        }
        
        // Compute voxel coordinates
        int cx = static_cast<int>((x - pc_range[0]) / voxel_size[0]);
        int cy = static_cast<int>((y - pc_range[1]) / voxel_size[1]);
        int cz = static_cast<int>((z - pc_range[2]) / voxel_size[2]);
        
        // Clamp to grid bounds
        cx = std::max(0, std::min(cx, grid_x - 1));
        cy = std::max(0, std::min(cy, grid_y - 1));
        cz = std::max(0, std::min(cz, grid_z - 1));
        
        // Coordinate key (stored as z, y, x for consistency with Python)
        std::array<int, 3> coord = {cz, cy, cx};
        
        auto it = coor_to_voxelidx.find(coord);
        int voxel_idx;
        
        if (it == coor_to_voxelidx.end()) {
            // New voxel
            if (voxel_data.num_voxels >= max_voxels) {
                continue;  // Skip if max voxels reached
            }
            
            voxel_idx = voxel_data.num_voxels;
            coor_to_voxelidx[coord] = voxel_idx;
            
            // Store coordinates (z, y, x)
            voxel_data.coors[voxel_idx * 3 + 0] = cz;
            voxel_data.coors[voxel_idx * 3 + 1] = cy;
            voxel_data.coors[voxel_idx * 3 + 2] = cx;
            
            voxel_data.num_voxels++;
        } else {
            voxel_idx = it->second;
        }
        
        // Add point to voxel
        int point_idx = voxel_data.num_points[voxel_idx];
        if (point_idx < max_points) {
            int offset = voxel_idx * max_points * num_features + point_idx * num_features;
            voxel_data.voxels[offset + 0] = x;
            voxel_data.voxels[offset + 1] = y;
            voxel_data.voxels[offset + 2] = z;
            voxel_data.voxels[offset + 3] = points[i * 5 + 3];  // intensity
            voxel_data.voxels[offset + 4] = points[i * 5 + 4];  // time_lag
            voxel_data.num_points[voxel_idx]++;
        }
    }
}

void create_pillar_input(const VoxelData& voxel_data,
                         const Config& config,
                         PillarInput& pillar_input) {
    int max_pillars = config.max_pillars;
    int max_points = config.max_points_per_pillar;
    int num_features = 10;  // x, y, z, intensity, time_lag, x_c, y_c, z_c, x_p, y_p
    
    const float* voxel_size = config.voxel_size.data();
    const float* pc_range = config.pc_range.data();
    int bev_w = config.grid_x;
    
    int num_pillars = std::min(voxel_data.num_voxels, max_pillars);
    pillar_input.num_pillars = num_pillars;
    
    // Initialize features: [1, 10, max_pillars, max_points]
    pillar_input.features.resize(num_features * max_pillars * max_points, 0.0f);
    
    // Initialize indices: [1, max_pillars, 2]
    pillar_input.indices.resize(max_pillars * 2, 0);
    for (int i = 0; i < max_pillars; ++i) {
        pillar_input.indices[i * 2 + 0] = 0;   // batch index
        pillar_input.indices[i * 2 + 1] = -1;  // invalid marker
    }
    
    // Process each pillar
    for (int i = 0; i < num_pillars; ++i) {
        int n_points = voxel_data.num_points[i];
        if (n_points == 0) continue;
        
        // Compute pillar center
        float x_sum = 0.0f, y_sum = 0.0f, z_sum = 0.0f;
        for (int j = 0; j < n_points; ++j) {
            int pt_offset = i * config.max_points_per_voxel * 5 + j * 5;
            x_sum += voxel_data.voxels[pt_offset + 0];
            y_sum += voxel_data.voxels[pt_offset + 1];
            z_sum += voxel_data.voxels[pt_offset + 2];
        }
        float x_center = x_sum / n_points;
        float y_center = y_sum / n_points;
        float z_center = z_sum / n_points;
        
        // Get pillar coordinate
        int cy = voxel_data.coors[i * 3 + 1];
        int cx = voxel_data.coors[i * 3 + 2];
        
        // Compute pillar position
        float x_pillar = cx * voxel_size[0] + pc_range[0] + voxel_size[0] / 2.0f;
        float y_pillar = cy * voxel_size[1] + pc_range[1] + voxel_size[1] / 2.0f;
        
        // Fill features for each point
        // Features layout: [10, max_pillars, max_points]
        // Channel order: x, y, z, intensity, time_lag, x_c, y_c, z_c, x_p, y_p
        for (int j = 0; j < n_points; ++j) {
            int pt_offset = i * config.max_points_per_voxel * 5 + j * 5;
            float x = voxel_data.voxels[pt_offset + 0];
            float y = voxel_data.voxels[pt_offset + 1];
            float z = voxel_data.voxels[pt_offset + 2];
            float intensity = voxel_data.voxels[pt_offset + 3];
            float time_lag = voxel_data.voxels[pt_offset + 4];
            
            // Store in [C, P, N] format (C=10, P=max_pillars, N=max_points)
            int base = i * max_points + j;
            pillar_input.features[0 * max_pillars * max_points + base] = x;
            pillar_input.features[1 * max_pillars * max_points + base] = y;
            pillar_input.features[2 * max_pillars * max_points + base] = z;
            pillar_input.features[3 * max_pillars * max_points + base] = intensity;
            pillar_input.features[4 * max_pillars * max_points + base] = time_lag;
            pillar_input.features[5 * max_pillars * max_points + base] = x - x_center;
            pillar_input.features[6 * max_pillars * max_points + base] = y - y_center;
            pillar_input.features[7 * max_pillars * max_points + base] = z - z_center;
            pillar_input.features[8 * max_pillars * max_points + base] = x - x_pillar;
            pillar_input.features[9 * max_pillars * max_points + base] = y - y_pillar;
        }
        
        // Compute BEV index
        pillar_input.indices[i * 2 + 1] = cy * bev_w + cx;
    }
}

void preprocess(const std::vector<float>& points,
                int num_points,
                const Config& config,
                std::vector<float>& features,
                std::vector<int>& indices) {
    // Step 1: Voxelization
    VoxelData voxel_data;
    points_to_voxels(points, num_points, config, voxel_data);
    
    // Step 2: Create pillar input
    PillarInput pillar_input;
    create_pillar_input(voxel_data, config, pillar_input);
    
    // Copy to output (add batch dimension)
    // features: [1, 10, max_pillars, max_points]
    features = std::move(pillar_input.features);
    
    // indices: [1, max_pillars, 2]
    indices = std::move(pillar_input.indices);
}

} // namespace centerpoint
