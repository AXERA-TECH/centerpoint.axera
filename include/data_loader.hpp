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
#include "centerpoint_common.hpp"

namespace centerpoint {

// Sample information
struct SampleInfo {
    std::string token;
    std::string points_path;
    std::string gt_path;
    int num_points;
};

// Load sample index from JSON
bool load_sample_index(const std::string& data_dir, std::vector<SampleInfo>& samples);

// Load point cloud from binary file
bool load_points(const std::string& path, std::vector<float>& points, int& num_points);

// Load configuration from JSON
bool load_config(const std::string& config_path, Config& config);

} // namespace centerpoint
