#pragma once

#include <iostream>             // Terminal IO
#include <fstream>
#include <mutex>
#include <filesystem>

#include <Eigen/Geometry>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

#include "../include/transformation_helpers.h"

namespace fs = std::filesystem;

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
    };

    struct RectifyMaps {
        cv::Mat depth_map_x;
        cv::Mat depth_map_y;
        cv::Mat color_map_x;
        cv::Mat color_map_y;
    };

    static void print_raw_calibration(k4a_calibration_t&);
    RectifyMaps process_calibration(k4a_calibration_t, fs::path);

    class K4AFrameExtractor {
        public:
            K4AFrameExtractor(std::string, std::string, std::string, ExportConfig);
            ~K4AFrameExtractor();

            void next_capture();
            uint8_t get_fps();

            int process_depth(k4a::image, int);
            int process_color(k4a::image, int);
            int process_ir(k4a::image, int);
            void process_rgbd(k4a::image, k4a::image, int);
            void process_pointcloud(k4a::image, k4a::image, int);
            void process_pose(k4a::image, k4a::image, int);
            void compute_undistortion_intrinsics();
            void record_timestamps(k4a::image, k4a::image, int);
            void extract_frames(int);
            void seek(int);

            std::string m_name;
            std::chrono::microseconds m_last_color_ts;
            std::chrono::microseconds m_last_depth_ts;
            std::mutex m_worker_lock;
            Eigen::Affine3f m_extrinsics = Eigen::Affine3f::Identity();

        protected:
            k4a::playback m_dev;
            k4a::capture m_capture{NULL};
            k4a_record_configuration_t m_dev_config;
            k4a_calibration_t m_calibration;
            const fs::path m_input_filename;
            const fs::path m_output_directory;
            std::ostringstream m_tsss;
            fs::path m_timestamp_path;
            std::ofstream m_timestamp_file;
            ExportConfig m_export_config;
            RectifyMaps m_rectify_maps;
            std::chrono::microseconds m_timestamp_offset;
    };
}
