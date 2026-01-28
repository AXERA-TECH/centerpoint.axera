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
#include "data_loader.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

namespace centerpoint {

bool load_sample_index(const std::string& data_dir, std::vector<SampleInfo>& samples) {
    std::string index_path = join_path(data_dir, "sample_index.json");
    
    std::ifstream file(index_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open sample index: " << index_path << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    file.close();
    
    // Parse samples array
    size_t samples_pos = json_str.find("\"samples\"");
    if (samples_pos == std::string::npos) {
        std::cerr << "No 'samples' key found in index file" << std::endl;
        return false;
    }
    
    size_t array_start = json_str.find("[", samples_pos);
    if (array_start == std::string::npos) return false;
    
    // Find each sample object
    size_t pos = array_start + 1;
    while (pos < json_str.length()) {
        size_t obj_start = json_str.find("{", pos);
        if (obj_start == std::string::npos) break;
        
        size_t obj_end = json_str.find("}", obj_start);
        if (obj_end == std::string::npos) break;
        
        std::string obj_str = json_str.substr(obj_start, obj_end - obj_start + 1);
        
        SampleInfo sample;
        
        // Parse token
        size_t token_pos = obj_str.find("\"token\"");
        if (token_pos != std::string::npos) {
            size_t quote_start = obj_str.find("\"", token_pos + 7);
            size_t quote_end = obj_str.find("\"", quote_start + 1);
            if (quote_start != std::string::npos && quote_end != std::string::npos) {
                sample.token = obj_str.substr(quote_start + 1, quote_end - quote_start - 1);
            }
        }
        
        // Parse points_path
        size_t pts_pos = obj_str.find("\"points_path\"");
        if (pts_pos != std::string::npos) {
            size_t quote_start = obj_str.find("\"", pts_pos + 13);
            size_t quote_end = obj_str.find("\"", quote_start + 1);
            if (quote_start != std::string::npos && quote_end != std::string::npos) {
                sample.points_path = obj_str.substr(quote_start + 1, quote_end - quote_start - 1);
            }
        }
        
        // Parse gt_path
        size_t gt_pos = obj_str.find("\"gt_path\"");
        if (gt_pos != std::string::npos) {
            size_t quote_start = obj_str.find("\"", gt_pos + 9);
            size_t quote_end = obj_str.find("\"", quote_start + 1);
            if (quote_start != std::string::npos && quote_end != std::string::npos) {
                sample.gt_path = obj_str.substr(quote_start + 1, quote_end - quote_start - 1);
            }
        }
        
        // Parse num_points
        size_t num_pos = obj_str.find("\"num_points\"");
        if (num_pos != std::string::npos) {
            size_t colon_pos = obj_str.find(":", num_pos);
            if (colon_pos != std::string::npos) {
                size_t num_start = colon_pos + 1;
                while (num_start < obj_str.length() && 
                       (obj_str[num_start] == ' ' || obj_str[num_start] == '\t')) {
                    num_start++;
                }
                size_t num_end = num_start;
                while (num_end < obj_str.length() && 
                       (obj_str[num_end] >= '0' && obj_str[num_end] <= '9')) {
                    num_end++;
                }
                if (num_end > num_start) {
                    sample.num_points = std::stoi(obj_str.substr(num_start, num_end - num_start));
                }
            }
        }
        
        if (!sample.token.empty()) {
            samples.push_back(sample);
        }
        
        pos = obj_end + 1;
    }
    
    std::cout << "[Data] " << samples.size() << " samples loaded" << std::endl;
    return !samples.empty();
}

bool load_points(const std::string& path, std::vector<float>& points, int& num_points) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open points file: " << path << std::endl;
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Calculate number of points (5 floats per point: x, y, z, intensity, time_lag)
    size_t num_floats = file_size / sizeof(float);
    num_points = num_floats / 5;
    
    points.resize(num_floats);
    file.read(reinterpret_cast<char*>(points.data()), file_size);
    file.close();
    
    return true;
}

bool load_config(const std::string& config_path, Config& config) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    file.close();
    
    SimpleJsonParser parser;
    parser.parse(json_str);
    
    // Parse voxel_generator config
    std::string voxel_gen_str = parser.get_string("voxel_generator", "");
    if (!voxel_gen_str.empty()) {
        SimpleJsonParser voxel_parser;
        voxel_parser.parse(voxel_gen_str);
        
        auto voxel_size = voxel_parser.get_float_array("voxel_size");
        if (voxel_size.size() >= 3) {
            config.voxel_size = {voxel_size[0], voxel_size[1], voxel_size[2]};
        }
        
        auto range = voxel_parser.get_float_array("range");
        if (range.size() >= 6) {
            config.pc_range = {range[0], range[1], range[2], range[3], range[4], range[5]};
        }
        
        config.max_points_per_voxel = voxel_parser.get_int("max_points_in_voxel", 20);
        
        auto max_voxel_num = voxel_parser.get_int_array("max_voxel_num");
        if (!max_voxel_num.empty()) {
            config.max_voxels = max_voxel_num.back();  // Use the larger value (test value)
        }
    }
    
    // Parse test_cfg
    std::string test_cfg_str = parser.get_string("test_cfg", "");
    if (!test_cfg_str.empty()) {
        SimpleJsonParser test_parser;
        test_parser.parse(test_cfg_str);
        
        auto post_range = test_parser.get_float_array("post_center_limit_range");
        if (post_range.size() >= 6) {
            config.post_center_limit_range = {post_range[0], post_range[1], post_range[2],
                                               post_range[3], post_range[4], post_range[5]};
        }
        
        config.score_threshold = test_parser.get_float("score_threshold", 0.1f);
        config.max_per_sample = test_parser.get_int("max_per_img", 500);
        config.out_size_factor = test_parser.get_int("out_size_factor", 4);
        
        // Parse NMS config
        std::string nms_str = test_parser.get_string("nms", "");
        if (!nms_str.empty()) {
            SimpleJsonParser nms_parser;
            nms_parser.parse(nms_str);
            config.nms_iou_threshold = nms_parser.get_float("nms_iou_threshold", 0.2f);
        }
    }
    
    // Compute grid size
    config.grid_x = static_cast<int>((config.pc_range[3] - config.pc_range[0]) / config.voxel_size[0]);
    config.grid_y = static_cast<int>((config.pc_range[4] - config.pc_range[1]) / config.voxel_size[1]);
    config.grid_z = static_cast<int>((config.pc_range[5] - config.pc_range[2]) / config.voxel_size[2]);
    
    // Compute BEV output size
    config.bev_h = config.grid_y / config.out_size_factor;
    config.bev_w = config.grid_x / config.out_size_factor;
    
    std::cout << "[Config] BEV: " << config.bev_w << "x" << config.bev_h 
              << ", voxels: " << config.max_voxels 
              << ", score_thr: " << config.score_threshold << std::endl;
    
    return true;
}

} // namespace centerpoint
