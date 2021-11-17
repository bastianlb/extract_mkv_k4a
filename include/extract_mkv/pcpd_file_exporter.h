#pragma once

#include <chrono>

#include "extract_mkv/filesystem.h"
#include "extract_mkv/utils.h"
#include "extract_mkv/extract_mkv_k4a.h"
#include "extract_mkv/kinect4azure_capture_wrapper.h"
#include "extract_mkv/timesync.h"

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
  enum DECODER_TYPE {
    COLOR,
    IR,
  };
  class H264Decoder {
    public:
        explicit H264Decoder(int, int, std::string, DECODER_TYPE);

        bool decode(std::vector<uint8_t>&, cv::Mat&);

    private:
      std::shared_ptr<NvDecoder> m_decoder;
      int m_width {0};
      int m_height {0};
      DECODER_TYPE m_decoder_type;
      std::string m_feed_name{};
      size_t m_fc{0};
  };
  class PCPDFileChannel {
    public:
      explicit PCPDFileChannel(fs::path, fs::path, std::string, ExportConfig);
      bool pcpd_extract_color(std::shared_ptr<KPU::Kinect4AzureCaptureWrapper>, cv::Mat&, bool write=false);
      bool pcpd_extract_depth(std::shared_ptr<KPU::Kinect4AzureCaptureWrapper>, bool write=false);
      bool pcpd_extract_infrared(std::shared_ptr<KPU::Kinect4AzureCaptureWrapper>, bool write=false);
      void load_mkv_info(std::string);
      void initialize();
      K4ADeviceWrapper m_device_wrapper{};
      std::string get_output_dir();
      std::string m_feed_name;
      std::chrono::microseconds m_last_depth_ts;
      int m_missing_frame_count;
      uint8_t m_recording_fps{30};

    protected:
      MkvTrackLoaderConfig m_trackloader_config{};
      std::shared_ptr<MkvSeekTrackLoader> m_loader;
      std::shared_ptr<H264Decoder> m_color_decoder;
      std::shared_ptr<H264Decoder> m_ir_decoder;
      ExportConfig m_export_config{};

      fs::path m_input_dir;
      fs::path m_output_dir;
      uint64_t m_frame_counter{1};
      uint16_t m_color_image_width{2048};
      uint16_t m_color_image_height{1536};
      uint16_t m_depth_image_width{640};
      uint16_t m_depth_image_height{576};
      pcpd::datatypes::PixelFormatType m_color_pixel_format{pcpd::datatypes::PixelFormatType::BGRA};
      k4a::calibration m_calibration;
  };

  class TimesynchronizerPCPD : public TimesynchronizerBase {
    public:
      explicit TimesynchronizerPCPD(ExportConfig &config);
      void run();
      void initialize_feeds(std::vector<fs::path>, fs::path);
      bool feed_forward(std::shared_ptr<PCPDFileChannel>);
      template <typename F>
      void run_threaded(const F* func, std::shared_ptr<PCPDFileChannel>, std::shared_ptr<KPU::Kinect4AzureCaptureWrapper>, int);
      int64_t get_frame_from_timestamp(int64_t);
      void monitor_frame_map();
      int m_max_frames_exported = std::numeric_limits<int>::max();

    protected:
      std::map<int, std::vector<std::shared_ptr<ProcessedData>>> m_frame_map;
      std::mutex m_frame_map_lock;
      std::vector<std::shared_ptr<PCPDFileChannel>> m_input_feeds;
      uint8_t m_recording_fps{30};
      fs::path m_output_dir;
      PCPDVideoWriter m_video_writer;
  };
}
