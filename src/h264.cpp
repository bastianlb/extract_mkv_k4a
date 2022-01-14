#include "extract_mkv/h264.h"
#include "extract_mkv/filesystem.h"

namespace extract_mkv {
  H264Decoder::H264Decoder(CUcontext &cu_context, int width, int height, std::string feed_name, DECODER_TYPE decoder_type)
            : m_width {width}
            , m_height {height}
            , m_decoder_type {decoder_type}
            , m_feed_name {feed_name} {
        cudaVideoCodec codec = cudaVideoCodec::cudaVideoCodec_H264;
 
        if (decoder_type == COLOR) {
          m_decoder = std::make_shared<NvDecoder>(cu_context, true, codec, true);
        } else if (decoder_type == IR) {
          m_decoder = std::make_shared<NvDecoder>(cu_context, true, codec, true);
        }

        if (!m_decoder) 
        {
          spdlog::error("Failed to create Nvenc Decoder - exiting.");
          return;
        }
        else
        {
          spdlog::info("Successfully created Decoder");
        }

        cudaStreamCreate(&m_cuda_stream);
  }

  H264Decoder::~H264Decoder() {
    cudaStreamDestroy(m_cuda_stream);
  }

  bool H264Decoder::decode(std::vector<uint8_t>& data_block, cv::cuda::GpuMat& image) {
      m_fc++;
      spdlog::trace("Calling NvDecoder::Decode with {0} bytes", data_block.size());
      int nFrameReturned = m_decoder->Decode(data_block.data(), data_block.size(), CUVID_PKT_ENDOFPICTURE, m_fc, m_cuda_stream);
      spdlog::trace("got {0} frames from decode", nFrameReturned);
      cudaStreamSynchronize(m_cuda_stream);

      if(nFrameReturned >= 1)
      {
          if (m_decoder->GetOutputFormat() != cudaVideoSurfaceFormat_NV12) 
          {
            spdlog::error("Unsupported videoSurfaceOutput format, only NV12 is supported.");
              return false;
          }

          int64_t timestamp;
          spdlog::trace("getting decoded frame ptr");
          const auto decodedFramePtr = m_decoder->GetFrame(&timestamp);

          cudaPointerAttributes attributes;
          std::string memory_type;
          if (cudaPointerGetAttributes(&attributes, decodedFramePtr) == cudaSuccess)
          {
              switch (attributes.type) {
                case cudaMemoryTypeHost:
                  memory_type = "Host"; break;
                case cudaMemoryTypeDevice:
                  memory_type = "Device"; break;
                case cudaMemoryTypeManaged:
                  memory_type = "Managed"; break;
                case cudaMemoryTypeUnregistered:
                  memory_type = "Unregistered"; break;
                default:
                  memory_type = "unknown";
              }
              spdlog::trace("Cuda memory type {0}", memory_type);
          }

          if (m_decoder_type == COLOR) {
            spdlog::trace("Decoding image frame of type color");
            Nv12ToColor32<BGRA32>(decodedFramePtr,m_width,(uint8_t*)image.data, sizeof(RGBA32)*m_width,
                                  m_width, m_height, 0, m_cuda_stream);
          } else if (m_decoder_type == IR) {
            spdlog::warn("GPU decode currently disabled for IR");
            return false;
            /*uint32_t stride_src = m_decoder->GetDeviceFramePitch();
            Nv12ToUint16(decodedFramePtr,stride_src,(uint8_t*)image.data,
                         sizeof(uint16_t)*m_width,
                         m_width,m_height,m_cuda_stream);
            */
            const size_t SIZE = m_decoder->GetDeviceFramePitch() * m_height;
            uint8_t* dev_decodedFramePtrDst;
            if (cudaMalloc((void**)&dev_decodedFramePtrDst, SIZE) != cudaSuccess)
            {
                std::cout << "cudaMalloc() for decodedFramePtrDst failed." << std::endl;
                return false;
            }
            uint32_t strideSrc = m_decoder->GetDeviceFramePitch();
            uint32_t strideDst = 2 * m_width;

            Nv12ToUint16(decodedFramePtr, strideSrc, dev_decodedFramePtrDst, strideDst, m_width, m_height, m_cuda_stream);

            //std::shared_ptr<uint8_t[]> img_data{ new uint8_t[SIZE] };
            if (cudaMemcpy((uint8_t *) image.data, dev_decodedFramePtrDst, SIZE, cudaMemcpyDeviceToDevice) != cudaSuccess)
            {
                std::cout << "cudaMemcpy() for decodedFramePtrDst failed." << std::endl;
                return false;
            }
            cudaFree(dev_decodedFramePtrDst);

            /*
            cv::Mat mat1(cv::Size(m_width, m_height), CV_16UC1, img_data.get(), strideDst);
            // cv::Mat mat(m_height, m_width, CV_16U, img_data.get());
            cv::Mat mat2(m_height, m_width, CV_8UC1);
            mat1.convertTo(mat2, CV_8UC1);
            auto path_template = "infrared_frame_{0}.jpg";
            fs::path fp = (fs::path{"/media/storage/atlas_export/"} / fmt::format(path_template, m_fc)).string();
            cv::imwrite(fp, mat2);
            */
          }
          if (cudaStreamSynchronize(m_cuda_stream) != cudaSuccess) {
            spdlog::warn("Failed to decode h264 image");
          };
          assert(!image.empty());
          return true;
      }
      else
      {
        spdlog::error("No Frames returned for feed {0}", m_feed_name);
        return false;
      }
  }

  bool H264Decoder::decode(std::vector<uint8_t>& data_block, cv::Mat& image) {
      // currently used for infrared image decode, no point of keeping on GPU
      m_fc++;
      spdlog::trace("Calling NvDecoder::Decode with {0} bytes", data_block.size());
      int nFrameReturned = m_decoder->Decode(data_block.data(), data_block.size(), CUVID_PKT_ENDOFPICTURE, m_fc, m_cuda_stream);
      spdlog::trace("got {0} frames from decode", nFrameReturned);
      cudaStreamSynchronize(m_cuda_stream);
      if(nFrameReturned >= 1)
      {
          if (m_decoder->GetOutputFormat() != cudaVideoSurfaceFormat_NV12) 
          {
            spdlog::error("Unsupported videoSurfaceOutput format, only NV12 is supported.");
              return false;
          }

          int64_t timestamp;
          spdlog::trace("getting decoded frame ptr");
          const auto decodedFramePtr = m_decoder->GetFrame(&timestamp);

          cudaPointerAttributes attributes;
          std::string memory_type;
          if (cudaPointerGetAttributes(&attributes, decodedFramePtr) == cudaSuccess)
          {
              switch (attributes.type) {
                case cudaMemoryTypeHost:
                  memory_type = "Host"; break;
                case cudaMemoryTypeDevice:
                  memory_type = "Device"; break;
                case cudaMemoryTypeManaged:
                  memory_type = "Managed"; break;
                case cudaMemoryTypeUnregistered:
                  memory_type = "Unregistered"; break;
                default:
                  memory_type = "unknown";
              }
              spdlog::trace("Cuda memory type {0}", memory_type);
          }
          const size_t SIZE = m_decoder->GetDeviceFramePitch() * m_height;
          uint8_t* dev_decodedFramePtrDst;
          if (cudaMalloc((void**)&dev_decodedFramePtrDst, SIZE) != cudaSuccess)
          {
              std::cout << "cudaMalloc() for decodedFramePtrDst failed." << std::endl;
              return false;
          }

          if (m_decoder_type == COLOR) {
            spdlog::trace("Decoding image frame of type color");
            Nv12ToColor32<BGRA32>(decodedFramePtr,m_width,(uint8_t*)image.data, sizeof(RGBA32)*m_width,
                                  m_width, m_height, 0, m_cuda_stream);
          } else if (m_decoder_type == IR) {
            uint32_t strideSrc = m_decoder->GetDeviceFramePitch();
            uint32_t strideDst = 2 * m_width;

            Nv12ToUint16(decodedFramePtr, strideSrc, dev_decodedFramePtrDst, strideDst, m_width, m_height);

            image.create(cv::Size(m_width, m_height), CV_16UC1);
            image.reserveBuffer(SIZE);
            if (cudaMemcpy((void *)image.data, dev_decodedFramePtrDst, SIZE, cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                std::cout << "cudaMemcpy() for decodedFramePtrDst failed." << std::endl;
                return false;
            }
            cudaFree(dev_decodedFramePtrDst);

            /*cv::Mat mat1(cv::Size(m_width, m_height), CV_16UC1, img_data.get(), strideDst);
            // cv::Mat mat(m_height, m_width, CV_16U, img_data.get());
            cv::Mat mat2(m_height, m_width, CV_8UC1);
            image.convertTo(mat2, CV_8UC1);
            auto path_template = "infrared_frame_{0}.jpg";
            fs::path fp = (fs::path{"/media/storage/atlas_export/"} / fmt::format(path_template, m_fc)).string();
            cv::imwrite(fp, mat2);
            */
          }
          cudaStreamSynchronize(m_cuda_stream);
          return true;
      }
      else
      {
        spdlog::error("No Frames returned for feed {0}", m_feed_name);
        return false;
      }
  }

}
