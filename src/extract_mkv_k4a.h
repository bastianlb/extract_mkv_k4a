#pragma once

#include <iostream>             // Terminal IO
#include <mutex>
#include <experimental/filesystem>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

namespace fs = std::experimental::filesystem;

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

    class K4AFrameExtractor {
        public:
            K4AFrameExtractor(std::string, std::string, std::string);
            ~K4AFrameExtractor();

            void next_capture();
            k4a::capture get_capture_handle();
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
    };
}
