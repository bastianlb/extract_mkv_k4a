// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include <Eigen/Geometry>
#include <k4a/k4a.h>
#include <opencv2/opencv.hpp>

struct color_point_t
{
    Eigen::Vector3f xyz;
    uint8_t rgb[3];
};

std::vector<color_point_t> image_to_pointcloud(const k4a_image_t point_cloud_image,
                                             const k4a_image_t color_image);

void tranformation_helpers_write_point_cloud(std::vector<color_point_t> points,
                                             const char *file_name);

cv::Size get_camera_depth_resolution(k4a_depth_mode_t depth_mode);

cv::Size get_camera_color_resolution(k4a_color_resolution_t color_mode);
