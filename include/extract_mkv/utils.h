#pragma once

#include <opencv2/opencv.hpp>

namespace extract_mkv {
    struct MissingDataException : public std::exception
    {
      const char * what () const throw ()
        {
            return "Failed to acquire data from the k4a_capture";
        }
    };

    struct ExportConfig {
      bool export_timestamp{false};
      bool export_color{false};
      bool export_depth{false};
      bool export_infrared{false};
      bool export_rgbd{false};
      bool export_pointcloud{false};
      bool align_clouds{false};
      bool export_extrinsics{false};
      bool export_bodypose{false};
      uint64_t start_ts{0};
      uint64_t end_ts{std::numeric_limits<std::uint64_t>::max()};
      size_t skip_frames{1};
    };

    struct RectifyMaps {
        cv::Mat depth_map_x;
        cv::Mat depth_map_y;
        cv::Mat color_map_x;
        cv::Mat color_map_y;
    };
}
