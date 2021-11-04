#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

#include "extract_mkv/pcpd_file_exporter.h"
#include "extract_mkv/kinect4azure_capture_wrapper.h"
#include "extract_mkv/extract_mkv_k4a.h"
#include "pcpd/processing/nvcodec/NvDecoder.h"
#include "pcpd/processing/nvcodec/ColorSpace.h"
#include "pcpd/processing/nvcodec/NvCodecUtils.h"
#include "pcpd/processing/cuda/detail/error_handling.cuh"
#include "pcpd/processing/cuda/detail/hardware.cuh"


using namespace pcpd;
using namespace pcpd::record;
using namespace rttr;

const int MIN_FILESIZE_BYTES = 1000000000;

std::string COLOR_TRACK_KEY = "COLOR";
std::string DEPTH_TRACK_KEY = "DEPTH";
std::string IR_TRACK_KEY = "INFRARED";

namespace extract_mkv {

  PCPDFileExporter::PCPDFileExporter(ExportConfig export_config) 
    : m_export_config(export_config){};

  void PCPDFileExporter::initialize_feeds(std::vector<fs::path> input_paths, fs::path output_directory) {
      for(auto input_dir : input_paths) {
          std::string feed_name = input_dir.parent_path().filename().string();
          spdlog::info("Initializing {0}", feed_name);
          // append the appropriate directory onto the output path, i.e. cn01 cn02 cn03..
          auto frame_extractor = std::make_shared<PCPDFileChannel>(input_dir, output_directory / feed_name, feed_name, m_export_config);
          m_input_feeds.push_back(frame_extractor);
      }

      spdlog::info("Finished initializing feeds, beginning frame export");
  };

  void PCPDFileExporter::run() {
      int i = 0;
      bool flag = true;
      while(i < 100 && flag) {
        for (auto ext : m_input_feeds) {
          flag = flag && ext->feed_forward();
        }
        ++i;
      }
  };

  PCPDFileChannel::PCPDFileChannel(fs::path input_dir, fs::path output_directory,
                                   std::string feed_name, ExportConfig export_config) :
                                  m_output_dir(output_directory), m_feed_name(feed_name),
                                  m_export_config(export_config) {
    // sets are sorted automatically
    std::set<std::string> filepaths;
    for (const auto & entry : fs::directory_iterator(input_dir))
        if (fs::file_size(entry) > MIN_FILESIZE_BYTES && entry.path().extension() == ".mkv")
            filepaths.insert(entry.path().string());


    fs::create_directories(m_output_dir);

    std::vector<std::string> fps{filepaths.begin(), filepaths.end()};
    if (fps.empty()) {
      spdlog::warn("No filepaths found for feed {0}", m_feed_name);
    } else {
      spdlog::info("Found filepaths for feed {0}", m_feed_name);
      for (const auto &fp: fps)
        spdlog::info(fp);
    }
    m_config.start_timestamp_offset_ns = export_config.start_ts;
    m_config.end_timestamp_offset_ns = export_config.end_ts;
    m_config.file_paths = fps;
    m_config.nth_frame = 1;
    // hack here... insert fake locator
    std::shared_ptr<service::Locator> spLocator = nullptr;
    m_loader = std::make_shared<MkvSeekTrackLoader>(spLocator, m_config);
    // TODO: need to get height/width of images.
    m_h264_decoder = std::make_shared<H264Decoder>(2048, 1536, m_feed_name);
    if (m_export_config.export_color)
      m_loader->addTrack(COLOR_TRACK_KEY);
    if (m_export_config.export_depth)
      m_loader->addTrack(DEPTH_TRACK_KEY);
    if (m_export_config.export_infrared)
      m_loader->addTrack(IR_TRACK_KEY);
    // read info from first filepath
    load_mkv_info(fps[0]);
  }

  void PCPDFileChannel::load_mkv_info(std::string fp) {
    MKVPlayerBase mkv{fp};
    auto ret1 = mkv.mkv_open();
    if(ret1 == PCPD_RESULT_FAILED)
    {
        spdlog::error(fmt::format("Could not open MKV file for schema inspection: {0}", fp));
        return;
    }
    size_t attachment_size = 2048;
    uint8_t attachment[attachment_size];
    const std::string RECORDING_SCHEMA_NAME = "record_schema.capnp";

    auto ret2 = mkv.mkv_get_attachment(RECORDING_SCHEMA_NAME.c_str(), attachment, &attachment_size);

    if(ret2 == PCPD_BUFFER_RESULT_TOO_SMALL)
    {
        spdlog::error(fmt::format("Buffer for recording schema attachment to small, need: {0}", attachment_size));
        return;
    }

    if(ret2 == PCPD_BUFFER_RESULT_FAILED)
    {
        spdlog::error(fmt::format("Could not find recording schema attachment: {0}", RECORDING_SCHEMA_NAME));
        return;
    }
    kj::ArrayPtr<kj::byte> bufferPtr = kj::arrayPtr(attachment, attachment_size);
    kj::ArrayInputStream ins (bufferPtr);
    ::capnp::InputStreamMessageReader message(ins);
    artekmed::schema::RecordingSchema::Reader reader = message.getRoot<artekmed::schema::RecordingSchema>();
    const std::string& mkv_name = reader.getName();
    pcpd::datatypes::DeviceCalibration device_calibration;
    KPU::deserializeCalibrations(reader, device_calibration);
    KPU::toK4A(device_calibration, m_calibration);
  }

  bool PCPDFileChannel::feed_forward() {
    bool flag = true;
    // create wrappers to get transformation / calibration handles
    K4AExportWrapper export_wrapper;
    KPU::Kinect4AzureCaptureWrapper k4a_wrapper{};
    export_wrapper.calibration = m_calibration;
    RectifyMaps rectify_maps = process_calibration(m_calibration, m_output_dir);
    export_wrapper.rectify_maps = rectify_maps;
    export_wrapper.capture = k4a_wrapper.capture_handle;
    if (m_export_config.export_color) {
      flag = flag && pcpd_extract_color(k4a_wrapper);
      process_color(export_wrapper.capture.get_color_image(),
                   export_wrapper,
                   m_output_dir,
                   m_frame_counter);
    }
    if (m_export_config.export_depth) {
      flag = flag && pcpd_extract_depth(k4a_wrapper);
      process_depth(export_wrapper.capture.get_depth_image(),
                   export_wrapper,
                   m_output_dir,
                   m_frame_counter);
    }
    if (m_export_config.export_infrared)
      flag = flag && pcpd_extract_infrared();


    if (m_export_config.export_rgbd) {
      process_rgbd(export_wrapper.capture.get_color_image(),
                   export_wrapper.capture.get_depth_image(),
                   export_wrapper,
                   m_output_dir,
                   m_frame_counter);
    }

    if (m_export_config.export_pointcloud) {
      process_pointcloud(export_wrapper.capture.get_color_image(),
                         export_wrapper.capture.get_depth_image(),
                         export_wrapper,
                         m_output_dir,
                         m_frame_counter);
    }

    spdlog::flush_on(spdlog::level::err);
    ++m_frame_counter;
    return flag;
  }

  bool PCPDFileChannel::pcpd_extract_color(KPU::Kinect4AzureCaptureWrapper &wrapper, bool write) {
    uint8 bits_per_element = 32;
    MkvDataBlock2 block;
    auto path_template = "color_frame_{0}.jpg";
    if(!m_loader->getNextDataBlock(block, COLOR_TRACK_KEY)) {
      spdlog::error("Unable to get next color block {0}..", m_frame_counter);
      return false;
    }

    cv::Mat color_image;
    bool ret = m_h264_decoder->decode(block.data->data_block, color_image);
    size_t stride = (color_image.cols * bits_per_element) / 8;
    size_t nbytes = stride * color_image.rows;
    ret &= wrapper.setColorImage(block.data->device_timestamp_ns,
                                 0, m_color_image_width,
                                 m_color_image_height, stride, (void*) (color_image.data),
                                 nbytes, m_color_pixel_format);
    if (!ret) {
      spdlog::error("Export failed for frame {0}", m_frame_counter);
    } else {
      if (write) {
        spdlog::info("Export color frame: {0} with size {1} and timestamp: {2}", m_frame_counter, color_image.size(), block.data->device_timestamp_ns);
        fs::path fp = (m_output_dir / fmt::format(path_template, m_frame_counter)).string();
        cv::imwrite(fp, color_image);
      }
    }
    return true;
  }

  bool PCPDFileChannel::pcpd_extract_depth(Kinect4AzureCaptureWrapper &wrapper, bool write) {
    uint8 bits_per_element = 16;
    MkvDataBlock2 block;
    auto path_template = "depth_frame_{0}.tiff";
    if(!m_loader->getNextDataBlock(block, DEPTH_TRACK_KEY)) {
      spdlog::error("Unable to get next color block {0}..", m_frame_counter);
      return false;
    }
    int width; 
    int height;

    std::vector<uint16_t> depth_out {};
    auto zdepth_compressor = std::make_shared<zdepth::DepthCompressor>();
    zdepth::DepthResult ret = zdepth_compressor->Decompress(block.data->data_block, width, height, depth_out);

    cv::Mat depth_image = cv::Mat(height, width, CV_16U, depth_out.data());
    size_t stride = (width * bits_per_element) / 8;
    size_t nbytes = stride * height;
    bool set_ret = wrapper.setDepthImage(block.data->device_timestamp_ns,
                          0, width, height, stride, (void*) (depth_image.data),
                          nbytes);

    if (!set_ret || ret != zdepth::DepthResult::Success) {
      spdlog::error("Export failed for frame {0}", m_frame_counter);
    } else {
      if (write) {
        spdlog::info("Export depth frame: {0} with size {1} and timestamp: {2}", 
            m_frame_counter, depth_image.size(), block.data->device_timestamp_ns);
        fs::path fp = (m_output_dir / fmt::format(path_template, m_frame_counter)).string();
        cv::imwrite(fp, depth_image);
      }
    }
    return true;
  }

  bool PCPDFileChannel::pcpd_extract_infrared() {
    throw cv::Exception();
  }

  H264Decoder::H264Decoder(int width, int height, std::string feed_name)
            : m_width {width}
            , m_height {height} 
            , m_feed_name {feed_name} {
        int gpu_id = pcpd::processing::cuda::gpuGetMaxGflopsDeviceId();

        pcpd::processing::cuda::initCudaDevice(gpu_id);

        spdlog::info("gpu-id: {0}", gpu_id);

        checkCudaErrors(cudaSetDevice(gpu_id));

        CUcontext context;
        checkCudaErrors(cuCtxGetCurrent(&context));
        cudaVideoCodec codec = cudaVideoCodec::cudaVideoCodec_H264;
 
        m_decoder = std::make_shared<NvDecoder>(context, false, codec, true);

        if (!m_decoder) 
        {
          spdlog::error("Failed to create Nvenc Decoder - exiting.");
          return;
        }
        else
        {
          spdlog::info("Successfully created Decoder");
        }
  }

  bool H264Decoder::decode(std::vector<uint8_t>& data_block, cv::Mat& image) {
      m_fc++;
      int nFrameReturned = m_decoder->Decode(data_block.data(), data_block.size(), CUVID_PKT_ENDOFPICTURE, m_fc, 0);

      if(nFrameReturned >= 1)
      {
          if (m_decoder->GetOutputFormat() != cudaVideoSurfaceFormat_NV12) 
          {
            spdlog::error("Unsupported videoSurfaceOutput format, only NV12 is supported.");
              return false;
          }

          int64_t timestamp;
          const auto decodedFramePtr = m_decoder->GetFrame(&timestamp);

          cv::Mat picYV12 = cv::Mat(m_height * 3/2, m_width, CV_8UC1, decodedFramePtr);

          // TOOD: can we use this pixelformat?
          cv::cvtColor(picYV12, image, cv::COLOR_YUV2BGRA_NV21);
          return true;
      }
      else
      {
        spdlog::error("No Frames returned for feed {0}", m_feed_name);
        return false;
      }
  }
}
