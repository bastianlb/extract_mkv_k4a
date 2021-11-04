// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "happly.h"

#include "extract_mkv/transformation_helpers.h"

std::vector<color_point_t> image_to_pointcloud(const k4a_image_t point_cloud_image,
                                             const k4a_image_t color_image) {
    std::vector<color_point_t> points;
    int width = k4a_image_get_width_pixels(point_cloud_image);
    int height = k4a_image_get_height_pixels(color_image);
    points.reserve(width * height);

    int16_t *point_cloud_image_data = (int16_t *)(void *)k4a_image_get_buffer(point_cloud_image);
    uint8_t *color_image_data = k4a_image_get_buffer(color_image);

    for (int i = 0; i < width * height; i++)
    {
        color_point_t point;
        point.xyz[0] = (float)point_cloud_image_data[3 * i + 0] / 1000;
        point.xyz[1] = (float)point_cloud_image_data[3 * i + 1] / 1000;
        point.xyz[2] = (float)point_cloud_image_data[3 * i + 2] / 1000;
        if (point.xyz[2] == 0)
        {
            continue;
        }

        point.rgb[0] = color_image_data[4 * i + 0];
        point.rgb[1] = color_image_data[4 * i + 1];
        point.rgb[2] = color_image_data[4 * i + 2];
        uint8_t alpha = color_image_data[4 * i + 3];

        if (point.rgb[0] == 0 && point.rgb[1] == 0 && point.rgb[2] == 0 && alpha == 0)
        {
            continue;
        }

        points.push_back(point);
    }
    return points;
}

void tranformation_helpers_write_point_cloud(std::vector<color_point_t> points,
                                             const char *file_name)
{
    std::vector<std::array<double, 3>> vertex_pos;
    std::vector<std::array<unsigned char, 3>> vertex_col;
    for (size_t i = 0; i < points.size(); ++i)
    {
        // image data is BGR
        vertex_pos.push_back({points[i].xyz[0], points[i].xyz[1], points[i].xyz[2]});
        vertex_col.push_back({points[i].rgb[2], points[i].rgb[1], points[i].rgb[0]});
    }
    happly::PLYData plyOut;

    plyOut.addVertexPositions(vertex_pos);
    plyOut.addVertexColors(vertex_col);
    plyOut.write(file_name, happly::DataFormat::Binary);
}

cv::Size get_camera_depth_resolution(k4a_depth_mode_t depth_mode) {
    int width;
    int height;
    switch (depth_mode) {
        case K4A_DEPTH_MODE_OFF:
            width = 0; height = 0;
            break;
        case K4A_DEPTH_MODE_NFOV_2X2BINNED:
            width = 320; height = 288;
            break;
        case K4A_DEPTH_MODE_NFOV_UNBINNED:
            width = 640; height = 576;
            break;
        case K4A_DEPTH_MODE_WFOV_2X2BINNED:
            width = 512; height = 512;
            break;
        case K4A_DEPTH_MODE_WFOV_UNBINNED:
            width = 1024; height = 1024;
            break;
        case K4A_DEPTH_MODE_PASSIVE_IR:
            width = 1024; height = 1024;
            break;
    }
    return cv::Size(width, height);
}

cv::Size get_camera_color_resolution(k4a_color_resolution_t color_mode) {
    int width;
    int height;
    switch (color_mode) {
        case K4A_COLOR_RESOLUTION_OFF:
            width = 0; height = 0;
            break;
        case K4A_COLOR_RESOLUTION_720P:
            width = 1280; height = 720;
            break;
        case K4A_COLOR_RESOLUTION_1080P:
            width = 1920; height = 1080;
            break;
        case K4A_COLOR_RESOLUTION_1440P:
            width = 2560; height = 1440;
            break;
        case K4A_COLOR_RESOLUTION_1536P:
            width = 2048; height = 1536;
            break;
        case K4A_COLOR_RESOLUTION_2160P:
            width = 3840; height = 2160;
            break;
        case K4A_COLOR_RESOLUTION_3072P:
            width = 4096; height = 3072;
            break;
    }
    return cv::Size(width, height);
}
