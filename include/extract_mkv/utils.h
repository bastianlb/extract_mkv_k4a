#pragma once

#include <thread>
#include <spdlog/spdlog.h>
#include "spdlog/fmt/ostr.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace extract_mkv {
    using namespace std::chrono;
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
        bool export_color_video{false};
        bool timesync{true};
        uint64_t max_frames_exported{std::numeric_limits<std::uint64_t>::max()};
        nanoseconds start_ts{0};
        nanoseconds end_ts{(time_point<system_clock>::max()).time_since_epoch()};
        size_t skip_frames{1};
        bool process_color() {
            return export_color  || export_color_video || export_pointcloud || export_rgbd;
        }
        bool process_depth() {
            return export_depth || export_pointcloud || export_rgbd;

        }
        bool process_infrared() {
            return export_infrared || export_bodypose;
        }
        bool grouped_process() {
            // group frames for batch process
            return export_color_video;
        }

        friend std::ostream &operator<<(std::ostream &os, const ExportConfig &c) {
            return os << "[ExportConfig: " 
                << "timestamps=" << c.export_timestamp << "\n"
                << "color=" << c.export_color << "\n"
                << "depth=" << c.export_depth << "\n"
                << "IR=" << c.export_infrared << "\n"
                << "RGBD=" << c.export_rgbd << "\n"
                << "pointclouds=" << c.export_pointcloud << "\n"
                << "align_clouds=" << c.align_clouds << "\n"
                << "extrinsics=" << c.export_extrinsics << "\n"
                << "bodypose=" << c.export_bodypose << "\n"
                << "color_video=" << c.export_color_video<< "\n"
                << "timesync=" << c.timesync << "\n"
                << "timerange=[" << c.start_ts.count() << ", " << c.end_ts.count() << "]\n"
                << "skip_frames=" << c.skip_frames << "\n"
                "]";
        }
    };

    struct RectifyMaps {
        cv::Mat depth_map_x;
        cv::Mat depth_map_y;
        cv::Mat color_map_x;
        cv::Mat color_map_y;
    };

    class ProcessedData {
        public:
            explicit ProcessedData() = default;
            std::string feed_name;
            cv::Mat depth_image;
            cv::Mat ir_image;
            cv::cuda::GpuMat color_image;
            int frame_id;
            std::chrono::microseconds timestamp_us;
            std::mutex lock;
    };

    class DataGroup {
        public:
            explicit DataGroup() = default;
            int frame_id;
            std::map<int, std::shared_ptr<ProcessedData>> feed_data_map;
            std::mutex lock;

    };

    class PCPDVideoWriter {
        public:
            explicit PCPDVideoWriter() = default;
            explicit PCPDVideoWriter(int num_splits, std::string filename, cv::Size video_size) 
                    : m_num_splits(num_splits), m_video_size(video_size), m_filename(filename) {
                setenv("TZ", "/usr/share/zoneinfo/Europe/Berlin", 1); // POSIX-specific
             };

            bool initialize_writer() {
                spdlog::info("Initializing video for file {0}", m_filename);
                int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
                double fps = 25.0;
                return m_writer.open(m_filename, codec, fps, m_video_size, true);
            };

            void write_frames(std::map<int, cv::cuda::GpuMat>& frames, std::chrono::microseconds timestamp) {
                if (!m_writer.isOpened()) {
                    spdlog::error("Video file not open!");
                    return;
                }
                cv::cuda::GpuMat output(m_video_size, CV_8UC4);
                if (frames.size() == 0) {
                    spdlog::error("Writing frames != 4 not currently supported.");
                    return;
                }
                const int w = m_video_size.width;
                const int h = m_video_size.height;
                const int w2 = static_cast<int>(w/2);
                const int h2 = static_cast<int>(h/2);
                auto el = frames.begin();
                std::vector<std::thread> threads;
                while (el != frames.end()) {
                    cv::Mat dst;
                    cv::Rect crop;
                    const int feed_id = el->first;
                    switch (feed_id) {
                         case 1:
                            crop = cv::Rect(0, 0, w2, h2);
                            break;
                        case 2:
                            crop = cv::Rect(0, h2, w2, h2);
                            break;
                        case 3:
                            crop = cv::Rect(w2, 0, w2, h2);
                            break;
                        case 4:
                            crop = cv::Rect(w2, h2, w2, h2);
                            break;
                        default:
                            spdlog::error("Recieved incorrect feed {0} in write video", el->first);
                            continue;
                    }
                    spdlog::trace("Cropping region for feed {0}: {1}, {2}, w: {3} h: {4} of image with dim {5}, {6}",
                            feed_id, crop.x, crop.y, crop.width, crop.height, w, h);
                    assert(crop.width != 0 && crop.height != 0);
                    threads.push_back(std::thread([crop, &w2, &h2]
                                (const std::map<int, cv::cuda::GpuMat> &in_frames, cv::cuda::GpuMat &out_frame, const int feed_id) {
                        spdlog::debug("Thread copying to video frame for feed {0}..", feed_id);
                        // TODO: is this threadsafe? the regions shouldn't overlap..
                        //spdlog::info(0 <= crop.width && crop.x + crop.width <= out_frame.cols && 0 <= crop.y && 0 <= crop.height && crop.y + crop.height <= out_frame.rows
                        cv::cuda::resize(in_frames.at(feed_id), out_frame(crop), cv::Size(w2, h2), cv::INTER_LINEAR);
                    }, std::ref(frames), std::ref(output), feed_id));
                    //cv::resize(frames.at(feed_id), output(crop), cv::Size(w2, h2), cv::INTER_LINEAR);
                    ++el;
                }
                using namespace std::chrono;

                auto time_point = system_clock::time_point(timestamp);
                auto in_time_t = system_clock::to_time_t(time_point);

                std::stringstream ss;
                ss << std::put_time(std::gmtime(&in_time_t), "%m-%d %H:%M:%S,");
                // apparently put_time doesn't format milliseconds
                ss << fmt::format("{0:03d}", duration_cast<milliseconds>(timestamp).count() % 1000);
                ss << " UTC";

                //cv::imwrite(m_filename + std::to_string(m_frame_count) + ".jpg", output);
                for (auto &t : threads) {
                    if (t.joinable())
                        t.join();
                }
                cv::cuda::cvtColor(output, output, cv::COLOR_RGBA2RGB);
                cv::Mat out_frame_device(m_video_size, CV_8UC3);
                //cv::putText(output, std::to_string(timestamp.count()), cv::Point(w * 0.70, h * 0.9), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255,255,255), 2);
                output.download(out_frame_device);
                spdlog::debug("Write video frame for timestamp {0}", timestamp.count());
                cv::putText(out_frame_device, ss.str(), cv::Point(w * 0.62, h * 0.95), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255,255,255), 2);
                m_writer.write(out_frame_device);
                m_frame_count++;
            };
            ~PCPDVideoWriter() {
                // TODO: object being prematurely closed?
                m_writer.release();
            };

        protected:
            int m_frame_count{1};
            int m_num_splits;
            cv::Size m_video_size;
            std::string m_filename;
            cv::VideoWriter m_writer{};
    };
    template <class T>
    void wrapArrayInVector( T *sourceArray, size_t arraySize, std::vector<T, std::allocator<T> > &targetVector ) {
      typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl *vectorPtr =
        (typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl *)((void *) &targetVector);
      vectorPtr->_M_start = sourceArray;
      vectorPtr->_M_finish = vectorPtr->_M_end_of_storage = vectorPtr->_M_start + arraySize;
    }

    template <class T>
    void releaseVectorWrapper( std::vector<T, std::allocator<T> > &targetVector ) {
      typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl *vectorPtr =
            (typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl *)((void *) &targetVector);
      vectorPtr->_M_start = vectorPtr->_M_finish = vectorPtr->_M_end_of_storage = NULL;
    }
}
