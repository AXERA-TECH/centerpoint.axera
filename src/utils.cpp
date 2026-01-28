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
#include "utils.hpp"
#include <fstream>
#include <cmath>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>

namespace centerpoint {

bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string join_path(const std::string& dir, const std::string& filename) {
    if (dir.empty()) return filename;
    if (dir.back() == '/') return dir + filename;
    return dir + "/" + filename;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void print_progress_bar(size_t current, size_t total, const std::string& prefix, 
                        float elapsed_ms) {
    const int bar_width = 40;
    float progress = static_cast<float>(current) / static_cast<float>(total);
    int pos = static_cast<int>(bar_width * progress);
    
    float fps = (elapsed_ms > 0) ? (current * 1000.0f / elapsed_ms) : 0.0f;
    float eta_seconds = (fps > 0 && current < total) ? 
                        ((total - current) / fps) : 0.0f;
    
    int eta_min = static_cast<int>(eta_seconds) / 60;
    int eta_sec = static_cast<int>(eta_seconds) % 60;
    
    std::cout << "\r" << prefix << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "] %3d%% [%zu/%zu] %.1ffps, ETA: %02d:%02d",
             static_cast<int>(progress * 100), current, total, fps, eta_min, eta_sec);
    std::cout << buffer;
    
    if (current == total) {
        std::cout << std::endl;
    }
    std::cout.flush();
}

// SimpleJsonParser implementation
bool SimpleJsonParser::parse(const std::string& json_str) {
    json_content = json_str;
    return !json_str.empty();
}

bool SimpleJsonParser::has_key(const std::string& key) const {
    return json_content.find("\"" + key + "\"") != std::string::npos;
}

std::string SimpleJsonParser::find_value(const std::string& key) const {
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = json_content.find(search_key);
    if (key_pos == std::string::npos) return "";
    
    size_t colon_pos = json_content.find(":", key_pos + search_key.length());
    if (colon_pos == std::string::npos) return "";
    
    size_t value_start = colon_pos + 1;
    while (value_start < json_content.length() && 
           (json_content[value_start] == ' ' || json_content[value_start] == '\n' ||
            json_content[value_start] == '\t' || json_content[value_start] == '\r')) {
        value_start++;
    }
    
    if (value_start >= json_content.length()) return "";
    
    // Handle different value types
    char first_char = json_content[value_start];
    
    if (first_char == '"') {
        // String value
        size_t end_quote = json_content.find("\"", value_start + 1);
        if (end_quote != std::string::npos) {
            return json_content.substr(value_start + 1, end_quote - value_start - 1);
        }
    } else if (first_char == '[') {
        // Array value
        int bracket_count = 1;
        size_t end_pos = value_start + 1;
        while (end_pos < json_content.length() && bracket_count > 0) {
            if (json_content[end_pos] == '[') bracket_count++;
            else if (json_content[end_pos] == ']') bracket_count--;
            end_pos++;
        }
        return json_content.substr(value_start, end_pos - value_start);
    } else if (first_char == '{') {
        // Object value
        int brace_count = 1;
        size_t end_pos = value_start + 1;
        while (end_pos < json_content.length() && brace_count > 0) {
            if (json_content[end_pos] == '{') brace_count++;
            else if (json_content[end_pos] == '}') brace_count--;
            end_pos++;
        }
        return json_content.substr(value_start, end_pos - value_start);
    } else {
        // Number or boolean
        size_t end_pos = value_start;
        while (end_pos < json_content.length() && 
               json_content[end_pos] != ',' && json_content[end_pos] != '}' &&
               json_content[end_pos] != ']' && json_content[end_pos] != '\n') {
            end_pos++;
        }
        std::string value = json_content.substr(value_start, end_pos - value_start);
        // Trim whitespace
        while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || 
                                   value.back() == '\r')) {
            value.pop_back();
        }
        return value;
    }
    
    return "";
}

float SimpleJsonParser::get_float(const std::string& key, float default_val) const {
    std::string value = find_value(key);
    if (value.empty()) return default_val;
    try {
        return std::stof(value);
    } catch (...) {
        return default_val;
    }
}

int SimpleJsonParser::get_int(const std::string& key, int default_val) const {
    std::string value = find_value(key);
    if (value.empty()) return default_val;
    try {
        return std::stoi(value);
    } catch (...) {
        return default_val;
    }
}

std::string SimpleJsonParser::get_string(const std::string& key, 
                                         const std::string& default_val) const {
    std::string value = find_value(key);
    if (value.empty()) return default_val;
    return value;
}

std::vector<float> SimpleJsonParser::get_float_array(const std::string& key) const {
    std::vector<float> result;
    std::string value = find_value(key);
    if (value.empty() || value[0] != '[') return result;
    
    // Remove brackets
    value = value.substr(1, value.length() - 2);
    
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        size_t start = item.find_first_not_of(" \t\n\r");
        size_t end = item.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            item = item.substr(start, end - start + 1);
        }
        try {
            result.push_back(std::stof(item));
        } catch (...) {}
    }
    
    return result;
}

std::vector<int> SimpleJsonParser::get_int_array(const std::string& key) const {
    std::vector<int> result;
    std::string value = find_value(key);
    if (value.empty() || value[0] != '[') return result;
    
    // Remove brackets
    value = value.substr(1, value.length() - 2);
    
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        size_t start = item.find_first_not_of(" \t\n\r");
        size_t end = item.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            item = item.substr(start, end - start + 1);
        }
        try {
            result.push_back(std::stoi(item));
        } catch (...) {}
    }
    
    return result;
}

} // namespace centerpoint
