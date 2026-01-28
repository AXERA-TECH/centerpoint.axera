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
#include "visualization.hpp"
#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <dirent.h>
#include <opencv2/core/utils/logger.hpp>

namespace centerpoint {

cv::Mat visualize_bev(const std::vector<float>& points,
                      int num_points,
                      const DetectionResult& result,
                      const Config& config,
                      int frame_idx,
                      float eval_range,
                      float conf_th) {
    // Image size and scale
    int img_size = 800;
    float scale = img_size / (2.0f * eval_range);
    int center = img_size / 2;
    
    // Create black background
    cv::Mat img(img_size, img_size, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Draw point cloud
    for (int i = 0; i < num_points; ++i) {
        float x = points[i * 5 + 0];
        float y = points[i * 5 + 1];
        
        // Filter points within range
        if (std::abs(x) >= eval_range || std::abs(y) >= eval_range) continue;
        
        // Remove close points (ego vehicle area)
        if (std::abs(x) < 3.0f && std::abs(y) < 3.0f) continue;
        
        // Convert to image coordinates
        int px = static_cast<int>(center + x * scale);
        int py = static_cast<int>(center - y * scale);
        
        if (px < 0 || px >= img_size || py < 0 || py >= img_size) continue;
        
        // Color by distance (viridis-like: purple -> cyan -> yellow)
        float dist = std::sqrt(x * x + y * y);
        float t = std::min(1.0f, dist / eval_range);
        
        int r, g, b;
        if (t < 0.5f) {
            r = static_cast<int>(68 + t * 2 * (49 - 68));
            g = static_cast<int>(1 + t * 2 * (104 - 1));
            b = static_cast<int>(84 + t * 2 * (142 - 84));
        } else {
            r = static_cast<int>(49 + (t - 0.5f) * 2 * (253 - 49));
            g = static_cast<int>(104 + (t - 0.5f) * 2 * (231 - 104));
            b = static_cast<int>(142 + (t - 0.5f) * 2 * (37 - 142));
        }
        
        img.at<cv::Vec3b>(py, px) = cv::Vec3b(b, g, r);
    }
    
    // Count detections above threshold
    int num_detections = 0;
    for (const auto& det : result.detections) {
        if (det.score >= conf_th) num_detections++;
    }
    
    // Draw detection boxes
    for (const auto& det : result.detections) {
        if (det.score < conf_th) continue;
        
        int label = det.label;
        cv::Scalar color = (label >= 0 && label < static_cast<int>(CLASS_COLORS.size())) 
                          ? CLASS_COLORS[label] : cv::Scalar(255, 255, 255);
        
        // Convert center to image coordinates
        int cx = static_cast<int>(center + det.x * scale);
        int cy = static_cast<int>(center - det.y * scale);
        
        // Apply angle transformation
        float vis_theta = -det.yaw - M_PI / 2.0f;
        float cos_t = std::cos(vis_theta);
        float sin_t = std::sin(vis_theta);
        
        // Different shapes based on class
        if (label == 8) {  // pedestrian - circle
            int radius = std::max(3, static_cast<int>(std::max(det.w, det.l) * scale / 2));
            cv::circle(img, cv::Point(cx, cy), radius, color, 2);
            // Draw heading line
            int head_x = static_cast<int>(cx + radius * cos_t);
            int head_y = static_cast<int>(cy - radius * sin_t);
            cv::line(img, cv::Point(cx, cy), cv::Point(head_x, head_y), color, 2);
        } else if (label == 9) {  // traffic_cone - small triangle
            int size = std::max(4, static_cast<int>(std::max(det.w, det.l) * scale));
            std::vector<cv::Point> pts = {
                cv::Point(cx, cy - size),
                cv::Point(cx - size / 2, cy + size / 2),
                cv::Point(cx + size / 2, cy + size / 2)
            };
            cv::fillPoly(img, std::vector<std::vector<cv::Point>>{pts}, color);
        } else if (label == 5) {  // barrier - thin rectangle
            float corners[4][2] = {
                {det.l / 2, det.w / 4}, {det.l / 2, -det.w / 4},
                {-det.l / 2, -det.w / 4}, {-det.l / 2, det.w / 4}
            };
            std::vector<cv::Point> pts(4);
            for (int i = 0; i < 4; ++i) {
                float rx = corners[i][0] * cos_t - corners[i][1] * sin_t + det.x;
                float ry = corners[i][0] * sin_t + corners[i][1] * cos_t + det.y;
                pts[i] = cv::Point(static_cast<int>(center + rx * scale),
                                   static_cast<int>(center - ry * scale));
            }
            cv::fillPoly(img, std::vector<std::vector<cv::Point>>{pts}, color);
        } else if (label == 6 || label == 7) {  // motorcycle, bicycle
            float corners[4][2] = {
                {det.l / 2, det.w / 2}, {det.l / 2, -det.w / 2},
                {-det.l / 2, -det.w / 2}, {-det.l / 2, det.w / 2}
            };
            std::vector<cv::Point> pts(4);
            for (int i = 0; i < 4; ++i) {
                float rx = corners[i][0] * cos_t - corners[i][1] * sin_t + det.x;
                float ry = corners[i][0] * sin_t + corners[i][1] * cos_t + det.y;
                pts[i] = cv::Point(static_cast<int>(center + rx * scale),
                                   static_cast<int>(center - ry * scale));
            }
            cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, true, color, 2);
            cv::Point front_mid((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2);
            cv::arrowedLine(img, cv::Point(cx, cy), front_mid, color, 2, cv::LINE_AA, 0, 0.4);
        } else {  // car, truck, bus, etc. - standard box
            float corners[4][2] = {
                {det.l / 2, det.w / 2}, {det.l / 2, -det.w / 2},
                {-det.l / 2, -det.w / 2}, {-det.l / 2, det.w / 2}
            };
            std::vector<cv::Point> pts(4);
            for (int i = 0; i < 4; ++i) {
                float rx = corners[i][0] * cos_t - corners[i][1] * sin_t + det.x;
                float ry = corners[i][0] * sin_t + corners[i][1] * cos_t + det.y;
                pts[i] = cv::Point(static_cast<int>(center + rx * scale),
                                   static_cast<int>(center - ry * scale));
            }
            cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, true, color, 2);
            // Draw front indicator
            cv::Point front_mid((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2);
            cv::line(img, cv::Point(cx, cy), front_mid, color, 2);
        }
    }
    
    // Draw frame info
    char text[64];
    snprintf(text, sizeof(text), "Frame: %d", frame_idx);
    cv::putText(img, text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(255, 255, 255), 2);
    snprintf(text, sizeof(text), "Detections: %d", num_detections);
    cv::putText(img, text, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2);
    
    // Draw legend
    int legend_y = 80;
    for (size_t cls_id = 0; cls_id < CLASS_NAMES.size(); ++cls_id) {
        cv::Scalar color = CLASS_COLORS[cls_id];
        cv::rectangle(img, cv::Point(10, legend_y), cv::Point(25, legend_y + 12), color, -1);
        cv::putText(img, CLASS_NAMES[cls_id], cv::Point(30, legend_y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        legend_y += 18;
    }
    
    return img;
}

bool create_video_from_images(const std::string& image_dir,
                              const std::string& output_path,
                              std::string& actual_path,
                              int fps) {
    // Get sorted list of images
    std::vector<std::string> image_files;
    
    DIR* dir = opendir(image_dir.c_str());
    if (!dir) {
        return false;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.size() > 4) {
            std::string ext = filename.substr(filename.size() - 4);
            if (ext == ".png" || ext == ".jpg") {
                image_files.push_back(filename);
            }
        }
    }
    closedir(dir);
    
    if (image_files.empty()) {
        return false;
    }
    
    std::sort(image_files.begin(), image_files.end());
    
    // Read first image to get dimensions
    std::string first_path = join_path(image_dir, image_files[0]);
    cv::Mat first_img = cv::imread(first_path);
    if (first_img.empty()) {
        return false;
    }
    
    int width = first_img.cols;
    int height = first_img.rows;
    
    // Limit video size
    const int max_width = 1920;
    const int max_height = 1080;
    if (width > max_width || height > max_height) {
        float scale = std::min(static_cast<float>(max_width) / width,
                              static_cast<float>(max_height) / height);
        width = static_cast<int>(width * scale);
        height = static_cast<int>(height * scale);
    }
    
    // Use a simple numbered filename to avoid OpenCV CAP_IMAGES pattern detection
    // The error occurs because OpenCV tries to parse the filename as an image sequence
    actual_path = output_path.substr(0, output_path.rfind('.')) + "_video.avi";
    
    // Suppress OpenCV error messages temporarily
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    
    cv::VideoWriter writer;
    
    // Try MJPG (most reliable on embedded systems)
    writer.open(actual_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                fps, cv::Size(width, height), true);
    
    if (!writer.isOpened()) {
        // Try XVID
        writer.open(actual_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                   fps, cv::Size(width, height), true);
    }
    
    // Restore logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    if (!writer.isOpened()) {
        std::cerr << "Failed to create video writer" << std::endl;
        return false;
    }
    
    int frame_count = 0;
    for (const auto& filename : image_files) {
        std::string img_path = join_path(image_dir, filename);
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;
        
        if (img.cols != width || img.rows != height) {
            cv::resize(img, img, cv::Size(width, height));
        }
        writer.write(img);
        frame_count++;
    }
    
    writer.release();
    
    std::cout << "Video: " << frame_count << " frames -> " << actual_path << std::endl;
    
    return true;
}

} // namespace centerpoint
