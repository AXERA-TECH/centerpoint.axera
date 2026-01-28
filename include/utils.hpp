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

namespace centerpoint {

// File utilities
bool file_exists(const std::string& path);
std::string join_path(const std::string& dir, const std::string& filename);

// Math utilities
float sigmoid(float x);

// Progress bar
void print_progress_bar(size_t current, size_t total, const std::string& prefix = "", 
                        float elapsed_ms = 0.0f);

// Simple JSON parser (for reading config files)
class SimpleJsonParser {
public:
    bool parse(const std::string& json_str);
    
    bool has_key(const std::string& key) const;
    float get_float(const std::string& key, float default_val = 0.0f) const;
    int get_int(const std::string& key, int default_val = 0) const;
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    std::vector<float> get_float_array(const std::string& key) const;
    std::vector<int> get_int_array(const std::string& key) const;
    
private:
    std::string json_content;
    std::string find_value(const std::string& key) const;
};

} // namespace centerpoint
