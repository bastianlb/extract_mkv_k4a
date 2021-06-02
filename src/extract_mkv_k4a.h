#pragma once

#include <iostream>             // Terminal IO
#include <fstream>
#include <mutex>
#include <filesystem>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

namespace fs = std::filesystem;

namespace extract_mkv {

    static void print_calibration(k4a_calibration_t&);
    void save_calibration(k4a_calibration_t, fs::path);

    struct MissingDataException : public std::exception
    {
      const char * what () const throw ()
        {
            return "Failed to acquire data from the k4a_capture";
        }
    };

    struct ExportConfig {
      const bool export_timestamp{false};
      const bool export_color{false};
      const bool export_depth{false};
      const bool export_infrared{false};
      const bool export_rgbd{false};
      const bool export_pointcloud{false};
    };

    class K4AFrameExtractor {
        public:
            K4AFrameExtractor(std::string, std::string, std::string, ExportConfig);
            ~K4AFrameExtractor();

            void next_capture(int);
            uint8_t get_fps();

            int process_depth(int);
            int process_color(int);
            int process_ir(int);
            void process_rgbd(int);
            void process_pointcloud(int);

            std::string m_name;
            double m_last_color_ts;
            double m_last_depth_ts;
            std::mutex lock;

        private:
            k4a::playback m_dev;
            k4a::capture m_capture;
            k4a_record_configuration_t m_dev_config;
            k4a_calibration_t m_calibration;
            const fs::path m_input_filename;
            const fs::path m_output_directory;
            std::ostringstream m_tsss;
            fs::path m_timestamp_path;
            std::ofstream m_timestamp_file;
            ExportConfig m_export_config;
    };
}
