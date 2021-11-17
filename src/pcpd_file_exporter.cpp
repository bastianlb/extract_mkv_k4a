#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

#include "extract_mkv/pcpd_file_exporter.h"
#include "extract_mkv/kinect4azure_capture_wrapper.h"
#include "extract_mkv/timesync.h"
#include "extract_mkv/extract_mkv_k4a.h"
#include "pcpd/processing/nvcodec/NvDecoder.h"
#include "pcpd/processing/nvcodec/ColorSpace.h"
#include "pcpd/processing/nvcodec/NvCodecUtils.h"
#include "pcpd/processing/cuda/detail/error_handling.cuh"
#include "pcpd/processing/cuda/detail/hardware.cuh"
#include "pcpd/processing/cuda/detail/developer_tools.cuh"
#include "pcpd/util/cuda_device_scope.h"
#include "pcpd/processing/nvcodec/NvCodecUtils.h"
#include "pcpd/processing/nvcodec/ColorSpace.h"

using namespace pcpd;
using namespace pcpd::record;
using namespace rttr;

const int MIN_FILESIZE_BYTES = 1000*1000*1000;
// const int MIN_FILESIZE_BYTES = 100*1000*1000; // 100MB

std::string COLOR_TRACK_KEY = "COLOR";
std::string DEPTH_TRACK_KEY = "DEPTH";
std::string IR_TRACK_KEY = "INFRARED";

namespace extract_mkv {

  TimesynchronizerPCPD::TimesynchronizerPCPD(ExportConfig &export_config) 
    : TimesynchronizerBase(export_config) {};

  void TimesynchronizerPCPD::initialize_feeds(std::vector<fs::path> input_paths, fs::path output_directory) {
      std::string export_name;
      for(auto input_dir : input_paths) {
          // TODO: this is annoying.. as dir has to end in /. make more robust?
          std::string feed_name = input_dir.parent_path().filename().string();
          export_name = input_dir.parent_path().parent_path().filename().string();
          spdlog::info("Initializing {0}", feed_name);
          // append the appropriate directory onto the output path, i.e. cn01 cn02 cn03..
          auto frame_extractor = std::make_shared<PCPDFileChannel>(input_dir, output_directory / feed_name, feed_name, m_export_config);
          m_input_feeds.push_back(frame_extractor);
      }

      spdlog::info("Finished initializing feeds, beginning frame export");
      if (m_export_config.export_color_video) {
        m_video_writer = PCPDVideoWriter(input_paths.size(), (output_directory / (export_name + ".avi")).string(), cv::Size(1280, 720));
        if (!m_video_writer.initialize_writer()) {
            spdlog::error("Failed to initialize video writer.");
        };
      }
      // large sync window for pcpd... this is just to keep them roughly aligned
      int fps_mult = 10;
      float fps = 10 * (1 / static_cast<float>(m_input_feeds[0]->m_recording_fps)) * pow(10, 6);
      m_sync_window = std::chrono::microseconds(static_cast<uint64_t>(fps));
  };

  void TimesynchronizerPCPD::run() {
      // naive counter for successive failures
      m_is_running = true;
      m_monitor_thread = std::thread([=] () {
        monitor();
      });
      m_performance_thread = std::thread([=] () {
        performance_monitor();
      });

      while(m_frames_exported < 20 && m_is_running) {
        for (auto feed : m_input_feeds) {
          if (!feed_forward(feed))
            feed->m_missing_frame_count++;
          if (feed->m_missing_frame_count > 30) {
            // too many missing frames
            m_is_running = false;
          }
        }
        ++m_frames_exported;
        while (m_frames_exported > 10) {
          // fast forward until streams are in sync again
          // look at the feed that is furthest ahead, and sync others to that
          // timepoint if they are lagging behind.
          bool sync_cond{true};
          auto first_feed = *std::max_element(m_input_feeds.begin(), m_input_feeds.end(),
                [] (auto lhs, auto rhs) {
                return lhs->m_last_depth_ts< rhs->m_last_depth_ts;
          });
          for (auto feed : m_input_feeds) {
            auto diff = first_feed->m_last_depth_ts - feed->m_last_depth_ts;
            if (diff > m_sync_window) {
              spdlog::warn("Frame: {0} - Feed {1} out of sync at {2}. Feed {3} is ahead at {4}, fast forward..",
                  m_frames_exported, feed->m_feed_name, feed->m_last_depth_ts.count(), first_feed->m_feed_name, first_feed->m_last_depth_ts.count());
              sync_cond = false;
              feed_forward(feed);
            }
          }
          if (sync_cond)
            break;
        }
        monitor_frame_map();
      }
      spdlog::info("Finishing.. total {0} frames exported", m_frames_exported);
      shutdown();
  };

  template <typename F>
  void TimesynchronizerPCPD::run_threaded(const F* func,
        std::shared_ptr<PCPDFileChannel> feed, std::shared_ptr<KPU::Kinect4AzureCaptureWrapper> k4a_wrapper, const int frame_id) {
      m_sem.wait();
      std::scoped_lock<std::mutex> lock1(m_lock);
      spdlog::debug("Launching thread for frame {0}", frame_id);
      m_worker_threads.push_back(std::thread([&] {
            (*func)(feed, k4a_wrapper, frame_id);
            m_finished_threads.push_back(std::this_thread::get_id());
            m_sem.notify();
            m_wait_cv.notify_one();
      }));
  };

  bool TimesynchronizerPCPD::feed_forward(std::shared_ptr<PCPDFileChannel> feed) {
    using namespace std::chrono;
    // note: lifecycle is not managed by this function. They are stored and cleaned up later.
    auto k4a_wrapper = std::make_shared<KPU::Kinect4AzureCaptureWrapper>(feed->m_feed_name);
    assert(k4a_wrapper->capture_handle != nullptr);
    bool flag = true;
    // create wrappers to get transformation / calibration handles
    microseconds image_timestamp{-1};
    auto processed_data = std::make_shared<ProcessedData>();
    processed_data->feed_name = feed->m_feed_name;
    if (m_export_config.export_color  || m_export_config.export_color_video) {
      flag = flag && feed->pcpd_extract_color(k4a_wrapper, processed_data->color_image);
      image_timestamp = k4a_wrapper->capture_handle.get_color_image().get_device_timestamp();
      processed_data->timestamp_us = image_timestamp;
    }
    if (m_export_config.export_depth) {
      flag = flag && feed->pcpd_extract_depth(k4a_wrapper);
      if (image_timestamp != microseconds(-1)) {
        image_timestamp = k4a_wrapper->capture_handle.get_depth_image().get_device_timestamp();
      }
    }

    if (m_export_config.export_infrared) {
      flag = flag && feed->pcpd_extract_infrared(k4a_wrapper);
    }

    if (image_timestamp == microseconds(-1)) {
      spdlog::error("Color or depth must be exported, or no time information is present!");
      return false;
    }
    // spdlog::trace("Got timestamp {0} from feed {1}", image_timestamp.count(), feed->m_feed_name);

    int64_t frame_id = get_frame_from_timestamp(duration_cast<nanoseconds>(image_timestamp).count());

    feed->m_last_depth_ts = duration_cast<microseconds>(image_timestamp);

    if (frame_id < 0) {
      spdlog::warn("Frame export failed for frame {0}", m_frames_exported);
      return false;
    }
    // reset frame count, we recieved frames.
    feed->m_missing_frame_count = 0;
    spdlog::debug("Got frame id {0}", frame_id);
    k4a_wrapper->frame_id = frame_id;
    auto it = m_frame_map.find(frame_id);

    m_frame_map_lock.lock();
    m_frame_map[frame_id].push_back(processed_data);
    m_frame_map_lock.unlock();

    if (m_export_config.export_color) {
      //auto lambda = [=] (std::shared_ptr<PCPDFileChannel> feed,
      //      std::shared_ptr<KPU::Kinect4AzureCaptureWrapper> k4a_wrapper, const int frame_id) {
          process_color(k4a_wrapper->capture_handle.get_color_image(),
                    feed->m_device_wrapper,
                    feed->get_output_dir(),
                    frame_id);
      //    };
      //run_threaded(&lambda, feed, k4a_wrapper, frame_id);
    }

    if (m_export_config.export_depth) {
      //auto lambda = [=] (std::shared_ptr<PCPDFileChannel> feed, 
      //      std::shared_ptr<KPU::Kinect4AzureCaptureWrapper> k4a_wrapper, const int frame_id) {
          process_depth(k4a_wrapper->capture_handle.get_depth_image(),
                        feed->m_device_wrapper,
                        feed->get_output_dir(),
                        frame_id);
      //    };
      //run_threaded(&lambda, feed, k4a_wrapper, frame_id);
    }

    if (m_export_config.export_infrared) {
      process_ir(k4a_wrapper->capture_handle.get_ir_image(),
                 feed->m_device_wrapper,
                 feed->get_output_dir(),
                 frame_id);
    }

    if (m_export_config.export_rgbd) {
      process_rgbd(k4a_wrapper->capture_handle.get_color_image(),
                   k4a_wrapper->capture_handle.get_depth_image(),
                   feed->m_device_wrapper,
                   feed->get_output_dir(),
                   frame_id);
    }

    if (m_export_config.export_pointcloud) {
      process_pointcloud(k4a_wrapper->capture_handle.get_color_image(),
                         k4a_wrapper->capture_handle.get_depth_image(),
                         feed->m_device_wrapper,
                         feed->get_output_dir(),
                         frame_id);
    }

    if (m_export_config.export_bodypose) {
      process_pose(feed->m_device_wrapper, feed->get_output_dir(), frame_id);
    }

    return flag;
  };

  void TimesynchronizerPCPD::monitor_frame_map() {
    spdlog::debug("Printing frame map...");
    // map should be ordered, so we can process in this manner
    // TODO: make this buffer size configurable
    int MAX_FRAME_BUFFER_SIZE = 100;
    int map_size = m_frame_map.size();
    if (map_size < MAX_FRAME_BUFFER_SIZE)
      // allow the feeds some time to catch up..
      return;
    int i = 0;
    auto element = m_frame_map.cbegin();
    // always leave a buffer window of 30
    // need to independently monitor frames so they are all more or less in sync;
    while (element != m_frame_map.cend() && map_size - i > MAX_FRAME_BUFFER_SIZE) {
      // debug printing
      if (element->second.size() > 0) {
        spdlog::debug("Frame map for frame {0}", element->first);
        for (auto wrapper : element->second) {
          spdlog::debug(wrapper->feed_name);
        }
      }
      // TODO: wait some time until frame is completely processed. Maybe check
      // other feeds how far they are, and artificially keep them in sync..
      std::chrono::microseconds timestamp;
      if (m_export_config.export_color_video && element->second.size() == m_input_feeds.size()) {
        std::map<int, cv::Mat> color_images;
        for (auto wrapper : element->second) {
          if (!wrapper->color_image.empty()) {
            auto feed_name = wrapper->feed_name;
            // note, this relies on feed name being at least 2 chars,
            // and last two chars being feed id. Not robust.
            int feed_id = std::stoi(feed_name.c_str() + 4 - 2);
            color_images[feed_id] = wrapper->color_image;
            timestamp = wrapper->timestamp_us;
          };
        }
        //all images present
        m_video_writer.write_frames(color_images, timestamp);
        // standard associative-container erase idiom.
        // TODO: make thread safe
      }
      spdlog::debug("Removing iterated map element");
      // TODO: this removes all
      m_frame_map.erase(element++);
      i++;
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
    double ns_per_frame = std::pow(10, 9) / static_cast<double>(m_recording_fps);

    spdlog::info("Setting start time {0} and end time {1} for channel {2}", export_config.start_ts, export_config.end_ts, m_feed_name);
    // Subtract offset for decoder.. bgroup size should be 5. otherwise images come out with artifacts.
    m_trackloader_config.start_timestamp_offset_ns = export_config.start_ts - 5 * ns_per_frame;
    m_trackloader_config.end_timestamp_offset_ns = export_config.end_ts;
    m_trackloader_config.file_paths = fps;
    m_trackloader_config.nth_frame = 1;
    // hack here... insert fake locator
    std::shared_ptr<service::Locator> spLocator = nullptr;
    m_loader = std::make_shared<MkvSeekTrackLoader>(spLocator, m_trackloader_config);
    // TODO: need to get height/width of images.
    m_color_decoder = std::make_shared<H264Decoder>(m_color_image_width, m_color_image_height,
                                                    m_feed_name, DECODER_TYPE::COLOR);
    m_ir_decoder = std::make_shared<H264Decoder>(m_depth_image_width, m_depth_image_height,
                                                 m_feed_name, DECODER_TYPE::IR);
    if (m_export_config.export_color || m_export_config.export_color_video)
      m_loader->addTrack(COLOR_TRACK_KEY);
    if (m_export_config.export_depth)
      m_loader->addTrack(DEPTH_TRACK_KEY);
    if (m_export_config.export_infrared)
      m_loader->addTrack(IR_TRACK_KEY);
    // read info from first filepath
    load_mkv_info(fps[0]);
    // need to call this last.. once we have calibration, etc
    // NOTE: recording fps not configurable right now
    initialize();
    //print_raw_calibration(m_calibration);
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

  void PCPDFileChannel::initialize() {
    m_device_wrapper.calibration = m_calibration;
    RectifyMaps rectify_maps = process_calibration(m_calibration, m_output_dir);
    m_device_wrapper.rectify_maps = rectify_maps;
  }

  int64_t TimesynchronizerPCPD::get_frame_from_timestamp(const int64_t timestamp) {
    if (timestamp < m_export_config.start_ts || timestamp > m_export_config.end_ts) {
      spdlog::warn("Skipping frame, not in indicated time range. {0} < {1}", timestamp, m_export_config.start_ts);
      return -1;
    }
    double ns_per_frame = std::pow(10, 9) / static_cast<double>(m_recording_fps);
    return std::round(static_cast<double>(timestamp - m_export_config.start_ts) / ns_per_frame);
  }

  bool PCPDFileChannel::pcpd_extract_color(std::shared_ptr<KPU::Kinect4AzureCaptureWrapper> wrapper,  cv::Mat &color_image, bool write) { MkvDataBlock2 block;
    auto path_template = "color_frame_{0}.jpg";
    if(!m_loader->getNextDataBlock(block, COLOR_TRACK_KEY)) {
      spdlog::error("Unable to get next color block {0}..", m_frame_counter);
      return false;
    }

    spdlog::debug("Extracting color image from mkv");
    bool ret = m_color_decoder->decode(block.data->data_block, color_image);
    // cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);
    size_t stride = (color_image.cols * color_image.elemSize());
    size_t nbytes = stride * color_image.rows;
    ret &= wrapper->setColorImage(block.data->device_timestamp_ns,
                                  block.data->device_timestamp_ns, m_color_image_width,
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

  bool PCPDFileChannel::pcpd_extract_depth(std::shared_ptr<Kinect4AzureCaptureWrapper> wrapper, bool write) {
    uint8 bits_per_element = 16;
    MkvDataBlock2 block;
    auto path_template = "depth_frame_{0}.tiff";
    if(!m_loader->getNextDataBlock(block, DEPTH_TRACK_KEY)) {
      spdlog::error("Unable to get next color block {0}..", m_frame_counter);
      return false;
    }
    int width; 
    int height;

    spdlog::debug("Extracting depth image from mkv");
    std::vector<uint16_t> depth_out {};
    auto zdepth_compressor = std::make_shared<zdepth::DepthCompressor>();
    zdepth::DepthResult ret = zdepth_compressor->Decompress(block.data->data_block, width, height, depth_out);

    cv::Mat depth_image = cv::Mat(height, width, CV_16U, depth_out.data());
    size_t stride = (width * bits_per_element) / 8;
    size_t nbytes = stride * height;
    bool set_ret = wrapper->setDepthImage(block.data->device_timestamp_ns,
                          block.data->device_timestamp_ns, width, height, stride, (void*) (depth_image.data),
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

  bool PCPDFileChannel::pcpd_extract_infrared(std::shared_ptr<KPU::Kinect4AzureCaptureWrapper> wrapper, bool write) {
    uint8 bits_per_element = 16;
    MkvDataBlock2 block;
    auto path_template = "color_frame_{0}.jpg";
    if(!m_loader->getNextDataBlock(block, IR_TRACK_KEY)) {
      spdlog::error("Unable to get next infrared block {0}..", m_frame_counter);
      return false;
    }

    spdlog::debug("Extracting infrared image from mkv");
    cv::Mat infrared_image(cv::Size(m_depth_image_width, m_depth_image_height), CV_16UC1);
    bool ret = m_ir_decoder->decode(block.data->data_block, infrared_image);
    size_t stride = (infrared_image.cols * bits_per_element) / 8;
    size_t nbytes = stride * infrared_image.rows;
    // TODO: pcpd uses depth image dims for the IR image. Is this correct?
    ret &= wrapper->setInfraredImage(block.data->device_timestamp_ns,
                                     block.data->device_timestamp_ns, m_depth_image_width,
                                     m_depth_image_height, stride, (void*) (infrared_image.data),
                                     nbytes);
    if (!ret) {
      spdlog::error("Export failed for frame {0}", m_frame_counter);
    } else {
      if (write) {
        spdlog::info("Export color frame: {0} with size {1} and timestamp: {2}", m_frame_counter, infrared_image.size(), block.data->device_timestamp_ns);
        fs::path fp = (m_output_dir / fmt::format(path_template, m_frame_counter)).string();
        cv::imwrite(fp, infrared_image);
      }
    }
    return true;
  }

  H264Decoder::H264Decoder(int width, int height, std::string feed_name, DECODER_TYPE decoder_type)
            : m_width {width}
            , m_height {height} 
            , m_decoder_type {decoder_type}
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

          if (m_decoder_type == COLOR) {
            spdlog::trace("Decoding image frame of type color");
            const auto decodedFramePtr = m_decoder->GetFrame(&timestamp);
            cv::Mat picYV12 = cv::Mat(m_height * 3/2, m_width, CV_8UC1, decodedFramePtr);

            // TOOD: can we use this pixelformat?
            cv::cvtColor(picYV12, image, cv::COLOR_YUV2BGRA_NV12);
          } else if (m_decoder_type == IR) {
            spdlog::trace("Decoding image frame of type ir");
            const auto decodedFramePtr = m_decoder->GetFrame(&timestamp);
            Nv12ToUint8(decodedFramePtr,m_decoder->GetDeviceFramePitch(),(uint8_t*)image.data,
                sizeof(uint16_t)*m_width,
                m_width,m_height,nullptr);
          }
          return true;
      }
      else
      {
        spdlog::error("No Frames returned for feed {0}", m_feed_name);
        return false;
      }
  }

  std::string PCPDFileChannel::get_output_dir() {
    return m_output_dir;
  }
}
