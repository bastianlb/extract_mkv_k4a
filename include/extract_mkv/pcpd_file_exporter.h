#pragma once

#include "extract_mkv/filesystem.h"
#include "extract_mkv/utils.h"
#include "extract_mkv/kinect4azure_capture_wrapper.h"

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
#include "pcpd/processing/nvcodec/NvDecoder.h"

using namespace pcpd;
using namespace pcpd::record;
using namespace rttr;

using namespace KPU;

namespace extract_mkv {
  class H264Decoder {
    public:
        explicit H264Decoder(int, int, std::string);

        bool decode(std::vector<uint8_t>&, cv::Mat&);

    private:
      std::shared_ptr<NvDecoder> m_decoder;
      int m_width {0};
      int m_height {0};
      std::string m_feed_name{};
      size_t m_fc{0};
  };
  class PCPDFileChannel {
    public:
      explicit PCPDFileChannel(fs::path, fs::path, std::string, ExportConfig);
      bool feed_forward();
      bool pcpd_extract_color(KPU::Kinect4AzureCaptureWrapper&, bool write=false);
      bool pcpd_extract_depth(KPU::Kinect4AzureCaptureWrapper&, bool write=false);
      bool pcpd_extract_infrared();
      void load_mkv_info(std::string);
      
    private:
      MkvTrackLoaderConfig m_config;
      std::shared_ptr<MkvSeekTrackLoader> m_loader;
      std::shared_ptr<H264Decoder> m_h264_decoder;
      ExportConfig m_export_config;
      k4a::calibration m_calibration;

      std::string m_feed_name;
      fs::path m_input_dir;
      fs::path m_output_dir;
      int m_frame_counter{1};
      uint16 m_color_image_width{2048};
      uint16 m_color_image_height{1536};
      pcpd::datatypes::PixelFormatType m_color_pixel_format{pcpd::datatypes::PixelFormatType::BGRA};
  };

  class PCPDFileExporter { 
    public: 
      explicit PCPDFileExporter(ExportConfig);

      void initialize_feeds(std::vector<fs::path>, fs::path);
      void run();
    private:
      ExportConfig m_export_config;
      std::vector<std::shared_ptr<PCPDFileChannel>> m_input_feeds;
  };
}
