#pragma once

#include <chrono>

#include "extract_mkv/filesystem.h"
#include "extract_mkv/utils.h"
#include "extract_mkv/extract_mkv_k4a.h"
#include "extract_mkv/kinect4azure_capture_wrapper.h"
#include "extract_mkv/timesync.h"
#include "extract_mkv/h264.h"

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
  class PCPDFileChannel {
    public:
      explicit PCPDFileChannel(fs::path, fs::path, std::string, ExportConfig&, CUcontext&);
      bool pcpd_extract_color(cv::cuda::GpuMat&, std::chrono::microseconds&, bool write=false);
      bool pcpd_extract_depth(cv::Mat&, std::chrono::microseconds&, bool write=false);
      bool pcpd_extract_infrared(cv::Mat&, std::chrono::microseconds&, bool write=false);
      void load_mkv_info(std::string);
      void initialize();
      int get_feed_id() {
        // note, this relies on feed name being at least 2 chars,
        // and last two chars being feed id. Not robust.
        return std::stoi(m_feed_name.c_str() + 4 - 2);
      }
      std::shared_ptr<K4ADeviceWrapper> m_device_wrapper;
      std::string get_output_dir();
      std::string m_feed_name;
      std::chrono::nanoseconds m_last_depth_ts{0};
      int m_missing_frame_count{0};
      float m_recording_fps{29.97};
      uint16_t m_color_image_width{2048};
      uint16_t m_color_image_height{1536};
      uint16_t m_depth_image_width{640};
      uint16_t m_depth_image_height{576};
      pcpd::datatypes::PixelFormatType m_color_pixel_format{pcpd::datatypes::PixelFormatType::BGRA};
      void write_rigid_transform(pcpd::datatypes::RigidTransform&);
      void write_rigid_transform(Eigen::Matrix4f&);
      std::unique_ptr<ProcessedData> m_processed_data;
      std::atomic<uint64_t> m_frame_counter{1};
      K4ATransformationContext m_transformation;

    protected:
      MkvTrackLoaderConfig m_trackloader_config{};
      std::shared_ptr<MkvSeekTrackLoader> m_loader;
      std::shared_ptr<H264Decoder> m_color_decoder;
      std::shared_ptr<H264Decoder> m_ir_decoder;
      ExportConfig m_export_config{};
      std::shared_ptr<KPU::Kinect4AzureCaptureWrapper> m_capture_wrapper;

      fs::path m_input_dir;
      fs::path m_output_dir;
      k4a::calibration m_calibration;
      Eigen::Matrix4f m_extrinsics;
  };

  class TimesynchronizerPCPD : public TimesynchronizerBase {
    public:
      explicit TimesynchronizerPCPD(ExportConfig &config);
      void run();
      void initialize_feeds(std::vector<fs::path>, fs::path);
      void advance_feed(std::shared_ptr<PCPDFileChannel>);
      bool process_feed(std::shared_ptr<PCPDFileChannel>, std::shared_ptr<ProcessedData>, const int);
      int64_t get_frame_from_timestamp(std::chrono::nanoseconds);
      void monitor_frame_map(bool flush=false);
      int m_max_frames_exported = std::numeric_limits<int>::max();

    protected:
      std::map<int, std::unique_ptr<DataGroup>> m_frame_map;
      std::mutex m_frame_map_lock;
      std::vector<std::shared_ptr<PCPDFileChannel>> m_input_feeds;
      uint8_t m_recording_fps{30};
      fs::path m_output_dir;
      fs::path m_timestamp_path;
      PCPDVideoWriter m_video_writer;
      CUcontext m_cu_context;
  };
}
