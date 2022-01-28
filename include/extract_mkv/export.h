#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "extract_mkv/extract_mkv_k4a.h"
#include "extract_mkv/filesystem.h"

namespace extract_mkv {

    int process_color_raw(cv::Mat image, std::shared_ptr<K4ADeviceWrapper> device_wrapper, fs::path output_directory, int frame_counter) {
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(95);

        cv::Mat undistorted_image;
        cv::remap(image, undistorted_image, device_wrapper->rectify_maps.color_map_x,
                  device_wrapper->rectify_maps.color_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
        cv::Mat out_image;
        undistorted_image.copyTo(out_image);
        // cv::resize(undistorted_image, out_image, cv::Size(512, 384));
        std::ostringstream ss;
        spdlog::debug("Writing color image..");
        ss << std::setw(10) << std::setfill('0') << frame_counter << "_color.jpg";
        fs::path image_path = output_directory / ss.str();
        cv::imwrite(image_path, out_image, compression_params);
        return 0;
    }


}
