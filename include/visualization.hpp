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
#include <opencv2/opencv.hpp>
#include "centerpoint_common.hpp"

namespace centerpoint {

// BGR colors for each class
const std::vector<cv::Scalar> CLASS_COLORS = {
    cv::Scalar(255, 0, 0),      // car - blue
    cv::Scalar(0, 165, 255),    // truck - orange
    cv::Scalar(0, 0, 255),      // construction_vehicle - red
    cv::Scalar(0, 255, 255),    // bus - yellow
    cv::Scalar(128, 0, 128),    // trailer - purple
    cv::Scalar(255, 255, 0),    // barrier - cyan
    cv::Scalar(0, 0, 255),      // motorcycle - red
    cv::Scalar(0, 255, 0),      // bicycle - green
    cv::Scalar(255, 0, 255),    // pedestrian - magenta
    cv::Scalar(0, 255, 255),    // traffic_cone - yellow
};

// Visualize BEV (Bird's Eye View) with point cloud and detections
cv::Mat visualize_bev(const std::vector<float>& points,
                      int num_points,
                      const DetectionResult& result,
                      const Config& config,
                      int frame_idx = 0,
                      float eval_range = 35.0f,
                      float conf_th = 0.5f);

// Create video from images
bool create_video_from_images(const std::string& image_dir,
                              const std::string& output_path,
                              std::string& actual_path,
                              int fps = 10);

} // namespace centerpoint
