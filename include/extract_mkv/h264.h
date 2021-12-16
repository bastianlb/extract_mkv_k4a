#pragma once

#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include "pcpd/processing/nvcodec/NvDecoder.h"
#include "pcpd/processing/nvcodec/ColorSpace.h"
#include "pcpd/processing/nvcodec/NvCodecUtils.h"

namespace extract_mkv {
  enum DECODER_TYPE {
    COLOR,
    IR,
  };
  class H264Decoder {
    public:
        explicit H264Decoder(CUcontext&, int, int, std::string, DECODER_TYPE);
        ~H264Decoder();

        bool decode(std::vector<uint8_t>&, cv::cuda::GpuMat&);
        bool decode(std::vector<uint8_t>&, cv::Mat&);

    private:
      std::shared_ptr<NvDecoder> m_decoder;
      int m_width {0};
      int m_height {0};
      DECODER_TYPE m_decoder_type;
      std::string m_feed_name{};
      size_t m_fc{0};
      cudaStream_t m_cuda_stream;
  };
}
