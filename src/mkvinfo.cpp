#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>

#include "libyuv.h"
#include "turbojpeg.h"
#include "opencv2/opencv.hpp"
#include "zdepth.hpp"
#include "Eigen/Core"
#include "capnp/serialize.h"
#include "cppfs/fs.h"
#include "cppfs/FilePath.h"
#include "cppfs/FileHandle.h"
#include "serialization/capnproto_serialization.h"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "pcpd/record/mkv/matroska_read.h"
#include "pcpd/config.h"
#include "pcpd/rttr_registration.h"
#include "pcpd/record/enumerations.h"

#include "pcpd/service/locator.h"
#include "pcpd/record/mkv_single_filereader.h"
#include "pcpd/record/mkv_sync_track_loader.h"

#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc.hpp"
#include "pcpd/processing/nvcodec/NvDecoder.h"
#include "pcpd/processing/nvcodec/ColorSpace.h"
#include "pcpd/processing/nvcodec/NvCodecUtils.h"
#include "pcpd/processing/cuda/detail/error_handling.cuh"
#include "pcpd/processing/cuda/detail/hardware.cuh"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

using namespace boost::program_options;
using namespace boost::filesystem;

using namespace pcpd;
using namespace pcpd::record;
using namespace rttr;


class H264Decoder
{
public:
    explicit H264Decoder(int width, int height)
        : m_width {width}
        , m_height {height}
    {
        init();
    }

    void decode(std::vector<uint8_t>& data_block, const std::string& filePath)
    {
        m_fc++;
        int nFrameReturned = m_decoder->Decode(data_block.data(), data_block.size(), CUVID_PKT_ENDOFPICTURE, m_fc, 0);

        if(nFrameReturned >= 1)
        {
            if (m_decoder->GetOutputFormat() != cudaVideoSurfaceFormat_NV12) 
            {
                std::cout << "Unsupported videoSurfaceOutput format, only NV12 is supported." << std::endl;
                return;
            }

            int64_t timestamp;
            const auto decodedFramePtr = m_decoder->GetFrame(&timestamp);

            cv::Mat picYV12 = cv::Mat(m_height * 3/2, m_width, CV_8UC1, decodedFramePtr);
            cv::Mat picRGBA;

            cv::cvtColor(picYV12, picRGBA, CV_YUV2RGBA_NV21);
            cv::imwrite(filePath, picRGBA);
        }
        else
        {
            std::cout << "No Frames returned" << std::endl;
        }
    }

private:
    void init()
    {
        int gpu_id = pcpd::processing::cuda::gpuGetMaxGflopsDeviceId();

        pcpd::processing::cuda::initCudaDevice(gpu_id);

        std::cout << "gpu-id: " << gpu_id << std::endl;

        checkCudaErrors(cudaSetDevice(gpu_id));

        CUcontext context;
        checkCudaErrors(cuCtxGetCurrent(&context));
        cudaVideoCodec codec = cudaVideoCodec::cudaVideoCodec_H264;
 
        m_decoder = std::make_shared<NvDecoder>(context, false, codec, true);

        if (!m_decoder) 
        {
            std::cout << "Failed to create Nvenc Decoder - exiting." << std::endl;
            return;
        }
        else
        {
            std::cout << "Successfully created Decoder" << std::endl;
        }
    }

std::shared_ptr<NvDecoder> m_decoder;
int m_width {0};
int m_height {0};
size_t m_fc{0};
};

class ColorTrackExporter
{
public:
    explicit ColorTrackExporter(MkvTrackLoaderConfig& config, const std::string& export_file_path)
        : m_config {config}
        , EXPORT_FILE_PATH {export_file_path}
    {}

    bool run()
    {
        std::shared_ptr<service::Locator> spLocator = nullptr;
        MkvSeekTrackLoader loader {spLocator, m_config};
        
        const std::string TRACK_NAME = "TIMECODE";

        loader.addTrack(TRACK_NAME);

        int frame_cnt {1};


        MkvDataBlock2 block;

        if(!loader.getNextDataBlock(block, TRACK_NAME))
            return false;

        // std::cout << "Export color frame: " << frame_cnt << " with timestamp: " << block.data->device_timestamp_ns << std::endl;

        std::vector<uint8_t> midi = block.data->data_block;
        return !midi.empty();
    }

private:
    MkvTrackLoaderConfig m_config;
    const std::string EXPORT_FILE_PATH;
};

bool loadMKVFilePaths(const std::string& sourceDir, std::vector<std::string>& filePaths)
{
    path p {sourceDir};

    if(!exists(p) || !is_directory(p))
        return false;

    directory_iterator it_end;

    for(directory_iterator it{p}; it != it_end; ++it)
    {
        if(is_regular_file(it->path()))
        {
            std::string file_path = it->path().string();

            if(extension(file_path) == ".mkv")
            {
                filePaths.push_back(file_path);
            }
        }
    }

    return true;
}

/**
 * Reads a timecode csv file and returns the mkv files
 */
void read_csv(const std::string& csv, std::vector<std::string>& filepaths){

    // Create an input filestream
    std::ifstream myFile(csv);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    std::string val;

    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        // Extract each column name
        while(std::getline(ss, colname, ',')){
        }
    }

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Extract each integer
        while(ss >> val){
            
            // Add the current integer to the 'colIdx' column's values vector
            filepaths.push_back(val.substr(0, val.size() - 2));
            
        }
    }

    // Close file
    myFile.close();
}

bool loadMKVFilePathsFromCSV(const std::string& csv, std::vector<std::string>& filePaths) {
    path p{csv};

    if(!exists(p) || !is_regular_file(p)) {
        return false;
    }

    read_csv(csv, filePaths);

    return true;
}

int main(int argc, char* argv[])
{
    try
    {
        uint64_t max_timestamp = std::numeric_limits<uint64_t>::max();

        options_description desc{"Usage"};
        desc.add_options()
            ("help,h",                                                     "Show Help message.")
            ("csv,i",     value<std::string>()->required(),                "Directory containing MKV files.")
            ("dout,o",    value<std::string>()->required(),                "Export directory.")
            ("nth,n",     value<uint>()->default_value(1),                 "Select every nth frame.")
            ("delay,d",   value<uint>()->default_value(0),                 "Delay in ms after a single frame was read.")
            ("start,s",   value<uint64_t>()->default_value(0),             "Start timestamp over all MKV files.")
            ("end,e",     value<uint64_t>()->default_value(max_timestamp), "End timestamp over all MKV files.");

        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);

        if(vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }

        notify(vm);

        std::string csv = vm["csv"].as<std::string>();
        std::string dout = vm["dout"].as<std::string>();
        std::vector<std::string> file_paths {};

        if(loadMKVFilePathsFromCSV(csv, file_paths))
        {
            std::cout << "Reading MKVs: " << csv << std::endl;

            for(auto fp : file_paths)
                std::cout << fp << std::endl;

            std::cout << "Export frames to:  " << dout << std::endl;
        }
        else
        {
            std::cout << "No MKV files found in: " << csv << std::endl;
            return -1;
        }

        std::vector<bool> result = std::vector<bool>(file_paths.size());
   
        for (int i = 0; i < file_paths.size(); ++i) {
            auto fp = std::vector<std::string>{file_paths[i]};

            MkvTrackLoaderConfig config {};
            config.file_paths                   = fp;
            config.nth_frame                    = vm["nth"].as<uint>();
            config.start_timestamp_offset_ns    = vm["start"].as<uint64_t>();
            config.end_timestamp_offset_ns      = vm["end"].as<uint64_t>();
            config.frame_delay_ms               = vm["delay"].as<uint>();

            ColorTrackExporter exporter {config, dout};
            result[i] = exporter.run();


        }

        for(int i = 0; i < result.size(); ++i) {
            std::cout<<file_paths[i]<<","<<result[i]<<std::endl;
        }

    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}



// #include <ostream>

// #include <opencv2/opencv.hpp>

// #include <pcpd/datatypes.h>


// int main(int argc, char* argv[])
// {
//     cv::Mat color_image = cv::imread("../test_data/0000000001_color.jpg");
//     cv::Mat depth_image = cv::imread("../test_data/0000000001_depth.tiff");
//     std::cout << "loaded files.." << std::endl;
//     /*
//     auto k4a_wrapper = std::make_shared<KPU::Kinect4AzureCaptureWrapper>("test");
//     uint64_t depth_ts_ns = 12345667;
//     size_t stride = (color_image.cols * color_image.elemSize());
//     size_t nbytes = stride * color_image.rows;
//     bool ret = k4a_wrapper->setColorImage(depth_ts_ns,
//                                           depth_ts_ns, color_image.rows,
//                                           color_image.cols, stride,
//                                           (void*) (color_image.data),
//                                           nbytes, pcpd::datatypes::PixelFormatType::BGRA);
//     */




//     return 0;
// }
