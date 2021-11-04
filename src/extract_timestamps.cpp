#define FMT_HEADER_ONLY 
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <chrono>
#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <functional>
#include <memory>
#include <future>
#include "fmt/format.h"

#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc.hpp"
#include "pcpd/processing/nvcodec/NvDecoder.h"
#include "pcpd/processing/nvcodec/ColorSpace.h"
#include "pcpd/processing/nvcodec/NvCodecUtils.h"
#include "pcpd/processing/cuda/detail/error_handling.cuh"
#include "pcpd/processing/cuda/detail/hardware.cuh"
#include "pcpd/record/mkv/matroska_read.h"
#include "cppfs/fs.h"
#include "cppfs/FilePath.h"
#include "cppfs/FileHandle.h"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "libyuv.h"
#include "turbojpeg.h"
#include "zdepth.hpp"

using namespace pcpd;
using namespace pcpd::record;

class ChannelExport
{
public:
    ChannelExport(const std::string& track_name, const std::string& subdir)
        : track_name {track_name}
        , subdir {subdir}
    {}

    const std::string track_name;
    const std::string subdir;

    track_reader_t* track_reader {nullptr};
    bool is_eof {false};
    uint64_t timestamp {0};
    uint frame_nr {0};
};

class MkvExport
{
public:
    MkvExport(const std::string& file_path, const std::string& dir, const std::vector<ChannelExport>& channels)
        : file_path {file_path}
        , dir {dir}
        , channels {channels}
    {}

    const std::string file_path;
    const std::string dir;
    std::vector<ChannelExport> channels;
};

class CSVLine
{
public:
    CSVLine(const uint frame_nr, const std::vector<uint64_t>& timestamps)
        : frame_nr {frame_nr}
        , timestamps {timestamps}
    {}

    const uint frame_nr;
    const std::vector<uint64_t> timestamps;
};

void save_color_frame_ffmpeg(const MkvExport& mkv_export, const ChannelExport& channel, 
    std::shared_ptr<MKVPlaybackDataBlock> buffer)
{
    std::string s = fmt::format("{:05d}", channel.frame_nr); 

    const size_t size = buffer->data_block.size();

    const std::string out_filepath_base = fmt::format("{0}/{1}/{2}_{3:05d}_{4}", 
        mkv_export.dir, channel.subdir, channel.track_name, channel.frame_nr, channel.timestamp);

    const std::string out_filepath_bin = fmt::format("{0}.bin", out_filepath_base);
    const std::string out_filepath_jpg = fmt::format("{0}.jpg", out_filepath_base);

    std::ofstream outfile(out_filepath_bin, std::ios::binary);
    outfile.write((char*)buffer->data_block.data(), size);
    outfile.flush();
    outfile.close();

    const std::string cmd = fmt::format("ffmpeg -i {0} {1}; rm {0}", out_filepath_bin, out_filepath_jpg);
    system(cmd.c_str());
}

void save_depth_frame(const MkvExport& mkv_export, const ChannelExport& channel, 
    std::shared_ptr<MKVPlaybackDataBlock> buffer)
{
    int width; 
    int height;

    std::vector<uint16_t> depth_out {};
    auto zdepth_compressor = std::make_shared<zdepth::DepthCompressor>();
    zdepth_compressor->Decompress(buffer->data_block, width, height, depth_out);

    cv::Mat mat = cv::Mat(height, width, CV_16U, depth_out.data());

    const std::string out_filepath = fmt::format("{0}/{1}/{2}_{3:05d}_{4}.tiff", 
        mkv_export.dir, channel.subdir, channel.track_name, channel.frame_nr, channel.timestamp);

    cv::imwrite(out_filepath, mat);
}

void save_infrared_frame(const MkvExport& mkv_export, const ChannelExport& channel, 
    std::shared_ptr<MKVPlaybackDataBlock> buffer)
{
    const int width = 640;
    const int height = 576;
    const size_t stride = channel.track_reader->stride;
    const size_t size = buffer->data_block.size();

    cv::Mat mat{cv::Size(width, height), CV_16UC1, buffer->data_block.data(), stride};

    const std::string out_filepath = fmt::format("{0}/{1}/{2}_{3:05d}_{4}.tiff", 
        mkv_export.dir, channel.subdir, channel.track_name, channel.frame_nr, channel.timestamp);

    cv::imwrite(out_filepath, mat);
}

bool get_timestamp_offset(MKVPlayerBase& mkv, uint64_t& ts_offset_ns)
{
    const std::string TAG_NAME = "PCPD_START_OFFSET_NS";
    const size_t TAG_SIZE = 25;

    size_t buffer_size {TAG_SIZE};
    char buffer[buffer_size] = {0};

    auto ret = mkv.mkv_get_tag(TAG_NAME.c_str(), buffer, &buffer_size);

    if(ret == PCPD_BUFFER_RESULT_TOO_SMALL)
    {
        std::cout << fmt::format("Failed to read tag {0}, size of {1} was too small, should be {2}", 
            TAG_NAME, TAG_SIZE, buffer_size) << std::endl;

        return false;
    }

    if(ret == PCPD_BUFFER_RESULT_FAILED)
    {
        std::cout << fmt::format("Failed to read tag {0}.", TAG_NAME) << std::endl;
        return false;
    }

    ts_offset_ns = std::stoull(std::string{buffer, buffer_size});

    return true;
}

void test01()
{

    const uint MIN_FRAME = 0;
    const uint MAX_FRAMES = 200;
    const uint SKIP_FRAMES = 10;

    std::string work_dir = "/home/lennart/ownCloud/atlas/scene_export_2109/patient_in_2";

    std::vector<ChannelExport> channels 
    {
        //ChannelExport{"DEPTH", "depth"},
        // ChannelExport{"COLOR", "color"}
        ChannelExport{"INFRARED", "infrared"}
    };

    std::vector<MkvExport> mkv_exports 
    {
        //MkvExport{"/artekmed/recordings/03_animal_trials/210914_animal_trial_03/recordings/recordings_1409_recording_01/cn01/capture-000026.mkv",
        //    fmt::format("{0}/cn01", work_dir), channels},
        MkvExport{"/media/lennart/1.42.6-25556/03_animal_trials/210824_animal_trial_02/recordings/recordings_2408_recording_06/cn02/capture-000000.mkv",
            fmt::format("{0}/cn03", work_dir), channels},
        //MkvExport{"/artekmed/recordings/03_animal_trials/210914_animal_trial_03/recordings/recordings_1409_recording_01/cn04/capture-000022.mkv",
        //    fmt::format("{0}/cn04", work_dir), channels},
    };

    pcpd_result_t ret1;
    pcpd_buffer_result_t ret2;
    pcpd_stream_result_t ret3;

    for(auto& mkv_export : mkv_exports)
    {
        MKVPlayerBase mkv {mkv_export.file_path};
        ret1 = mkv.mkv_open();

        if(!PCPD_SUCCEEDED(ret1))
        {
            std::cout << "Failed to open mkv file." << std::endl;
            return;
        }

        uint64_t ts_offset_ns {0}; 
        if(!get_timestamp_offset(mkv, ts_offset_ns))
            return;

        std::vector<track_reader_t*> tracks {};

        for(auto& channel : mkv_export.channels)
        {
            track_reader_t* track_reader = mkv.get_track_reader_by_name(channel.track_name);

            if(!track_reader)
            {
                std::cout << "Failed to get track_reader for: " << channel.track_name << std::endl;
                return;
            }

            channel.track_reader = track_reader;
            channel.timestamp = ts_offset_ns;

            // Check directories or create them...

            std::vector<std::string> dir_paths 
            {
                work_dir,
                mkv_export.dir,
                fmt::format("{0}/{1}", mkv_export.dir, channel.subdir)
            };

            for(auto& dir_path : dir_paths)
            {
                auto dir = cppfs::fs::open(dir_path);

                if(!dir.exists() && !dir.isDirectory())
                {
                    if(!dir.createDirectory())
                    {
                        std::cout << "Failed to create directory: " << dir_path << std::endl;
                        return;
                    }
                }
            }
        }

        bool eof = false;
        std::vector<CSVLine> csv_export {};
        uint frame_nr {1};
        std::vector<uint64_t> timestamps {};

        while(!eof)
        {

            timestamps.clear();

            for(auto& channel : mkv_export.channels)
            {
                if(channel.is_eof)
                {
                    timestamps.push_back(0);
                    continue;
                }

                auto buffer = std::make_shared<MKVPlaybackDataBlock>();
                ret3 = mkv.get_data_block(channel.track_reader, buffer, true);

                if(ret3 == PCPD_STREAM_RESULT_EOF)
                {
                    channel.is_eof = true;
                    timestamps.push_back(0);
                    continue;
                }

                if(ret3 == PCPD_STREAM_RESULT_FAILED)
                {
                    std::cout << "Failed to read data from track:" << channel.track_name << std::endl;
                    return;
                }
                
                // Mkv Cluster gives us ts in usec so we have to *1000 to get ns.
                channel.timestamp = ts_offset_ns + buffer->device_timestamp_usec * 1000;
                channel.frame_nr = frame_nr;

                timestamps.push_back(channel.timestamp);

                eof = std::all_of(mkv_export.channels.begin(), mkv_export.channels.end(), 
                        [](auto& channel) -> bool { return channel.is_eof;});

                if (frame_nr < MIN_FRAME || frame_nr % SKIP_FRAMES != 0)
                    continue;

                if(channel.track_name == "COLOR")
                {
                    save_color_frame_ffmpeg(mkv_export, channel, buffer);
                }
                else if(channel.track_name == "DEPTH")
                {
                    save_depth_frame(mkv_export, channel, buffer);
                }
                else if(channel.track_name == "INFRARED")
                {
                    save_infrared_frame(mkv_export, channel, buffer);
                }
            }

            csv_export.push_back(CSVLine {frame_nr, timestamps});
            frame_nr += 1;

            if(frame_nr > MAX_FRAMES)
                break;
        }

        std::string outfile_path = fmt::format("{0}/timestamps.csv", mkv_export.dir);
        std::ofstream outfile(outfile_path);

        outfile << fmt::format("frame_nr,depth_timestamps,color_timestamps") << std::endl;
        for(auto& line : csv_export)
        {
            outfile << fmt::format("{0},{1},{2}", line.frame_nr, line.timestamps[0], line.timestamps[1]) << std::endl;
        }

        outfile.flush();
        outfile.close();
    }
}


int main(int argc, char* argv[])
{
    test01();

    return 0;
}
