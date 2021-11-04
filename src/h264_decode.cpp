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

using namespace pcpd;
using namespace pcpd::record;

void test01()
{
    const std::string h264_filepath {"/home/sutorj/mnt/sdc3/narvis/Atlas062021/take03/cn01/capture-000090.mkv"};

    MKVPlayerBase mkv{h264_filepath};
    
    auto result1 = mkv.mkv_open();

    if(!PCPD_SUCCEEDED(result1))
    {
        std::cout << "Failed to open mkv file." << std::endl;
        return;
    }
    else 
    {
        std::cout << "Opened mkv file" << std::endl;
    }

    track_reader_t* track_reader = mkv.get_track_reader_by_name("COLOR");

    if(!track_reader)
    {
        std::cout << "Failed to get track_reader." << std::endl;
        return;
    }
    else 
    {
        std::cout << "Opened track" << std::endl;
    }

    auto buffer = std::make_shared<MKVPlaybackDataBlock>();
    auto result2 = mkv.get_data_block(track_reader, buffer, true);

    if(result2 == PCPD_STREAM_RESULT_EOF || result2 == PCPD_STREAM_RESULT_FAILED)
    {
        std::cout << "Failed to read data from track." << std::endl;
        return;
    }
    else
    {
        std::cout << "Read data from track with size: " << buffer->data_block.size() << std::endl;
    }
    

    const size_t size = buffer->data_block.size();
    const std::string out_filepath {"debug_h264_blob_" + std::to_string(size) + ".bin"};
    std::ofstream outfile(out_filepath, std::ios::binary);
    outfile.write((char*)buffer->data_block.data(), size);
    outfile.flush();
    outfile.close();


    int gpu_id = pcpd::processing::cuda::gpuGetMaxGflopsDeviceId();

    pcpd::processing::cuda::initCudaDevice(gpu_id);

    std::cout << "gpu-id: " << gpu_id << std::endl;

    checkCudaErrors(cudaSetDevice(gpu_id));

    cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate ( &(stream) ));

    CUcontext context;
    checkCudaErrors(cuCtxGetCurrent(&context));
    cudaVideoCodec codec = cudaVideoCodec::cudaVideoCodec_H264;

    
    auto decoder = std::make_shared<NvDecoder>(context, false, codec, true);

    if (!decoder) 
    {
        std::cout << "Failed to create Nvenc Decoder - exiting." << std::endl;
        return;
    }
    else
    {
        std::cout << "Successfully created Decoder" << std::endl;
    }

    const uint8_t* compressedData = buffer->data_block.data();
    const size_t compressedDataSize = buffer->data_block.size();

    int width {2048};
    int height {1536};
    size_t fc {0};

    int nFrameReturned = decoder->Decode(buffer->data_block.data(), buffer->data_block.size(), CUVID_PKT_ENDOFPICTURE, fc++, 0);

    if(nFrameReturned >= 1)
    {
        std::cout << "nFrameReturned: " << nFrameReturned << std::endl;

        if (decoder->GetOutputFormat() != cudaVideoSurfaceFormat_NV12) 
        {
            std::cout << "Unsupported videoSurfaceOutput format, only NV12 is supported." << std::endl;
            return;
		}

        int64_t timestamp;
        const auto decodedFramePtr = decoder->GetFrame(&timestamp);

        cv::Mat picYV12 = cv::Mat(height * 3/2, width, CV_8UC1, decodedFramePtr);
        cv::Mat picRGBA;

        cv::cvtColor(picYV12, picRGBA, CV_YUV2RGBA_NV21);
        cv::imwrite("rgba_test.bmp", picRGBA);  //only for test
        cv::imwrite("rgba_test.jpg", picRGBA);

        // Run ffmpeg for comparison 
        system("for i in *.bin; do ffmpeg -i \"$i\"  \"${i%.*}.jpg\"; done");

    }
    else
    {
        std::cout << "No Frames returned" << std::endl;
    }


    if (stream) 
    {
        cudaStreamDestroy(stream);
    }

    std::cout << "DONE!" << std::endl;
}


int main(int argc, char* argv[])
{
    test01();
    return 0;
}
