#pragma once

#include <iostream>             // Terminal IO
#include <fstream>
#include <mutex>

#include <Eigen/Geometry>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

#include "extract_mkv/utils.h"
#include "extract_mkv/filesystem.h"
#include "extract_mkv/transformation_helpers.h"

namespace extract_mkv {

    void print_raw_calibration(k4a_calibration_t&);
    RectifyMaps process_calibration(k4a_calibration_t, fs::path);

    class K4AFrameExtractor {
        public:
            K4AFrameExtractor(std::string, std::string, std::string, ExportConfig);
            ~K4AFrameExtractor();

            void next_capture();
            uint8_t get_fps();

            void record_timestamps(k4a::image, k4a::image, int);
            void extract_frames(int);
            void seek(int);

            std::string m_name;
            std::chrono::microseconds m_last_color_ts;
            std::chrono::microseconds m_last_depth_ts;
            std::mutex m_worker_lock;
            Eigen::Affine3f m_extrinsics = Eigen::Affine3f::Identity();
            k4a::calibration m_calibration;

        protected:
            k4a::playback m_dev;
            k4a::capture m_capture{NULL};
            k4a_record_configuration_t m_dev_config;
            const fs::path m_input_filename;
            const fs::path m_output_directory;
            std::ostringstream m_tsss;
            fs::path m_timestamp_path;
            std::ofstream m_timestamp_file;
            ExportConfig m_export_config;
            RectifyMaps m_rectify_maps;
            std::chrono::microseconds m_timestamp_offset;
    };
    struct K4ADeviceWrapper {
        RectifyMaps rectify_maps;
        k4a::calibration calibration;
        Eigen::Affine3f m_extrinsics = Eigen::Affine3f::Identity();
    };
    int process_depth(k4a::image, std::shared_ptr<K4ADeviceWrapper>, fs::path, int);
    int process_color(k4a::image, std::shared_ptr<K4ADeviceWrapper>, fs::path, int);
    void process_ir(k4a::image, std::shared_ptr<K4ADeviceWrapper>, fs::path, int);
    void process_pointcloud(k4a::image, k4a::image, std::shared_ptr<K4ADeviceWrapper>, fs::path, int, bool align_clouds=false);
    void process_pose(std::shared_ptr<K4ADeviceWrapper>, fs::path, int);
    void compute_undistortion_intrinsics();
    class K4ATransformationContext {
        public:
            K4ATransformationContext() = default;
            void init_transformation(k4a::calibration);
            void process_rgbd(k4a::image, int, int, std::shared_ptr<K4ADeviceWrapper>, fs::path, int);
            k4a::transformation m_transformation;
    };
}
