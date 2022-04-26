#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <numeric>
#include <spdlog/spdlog.h>
#include <json/json.h>
#include <json/writer.h>

#include <opencv2/cudaarithm.hpp>
#include <Eigen/Dense>

#include "extract_mkv/pcpd_file_exporter.h"
#include "extract_mkv/kinect4azure_capture_wrapper.h"
#include "extract_mkv/timesync.h"
#include "extract_mkv/extract_mkv_k4a.h"
#include "extract_mkv/export.h"
#include "extract_mkv/transformation_helpers.h"
#include "pcpd/processing/cuda/detail/error_handling.cuh"
#include "pcpd/processing/cuda/detail/hardware.cuh"
#include "pcpd/processing/cuda/detail/developer_tools.cuh"
#include "pcpd/util/cuda_device_scope.h"
#include "pcpd/processing/nvcodec/NvCodecUtils.h"
#include "pcpd/processing/nvcodec/ColorSpace.h"
#include "pcpd/processing/nvcodec/NvDecoder.h"

using namespace pcpd;
using namespace pcpd::record;
using namespace rttr;

const int MIN_FILESIZE_BYTES = 1300*1000*1000; // 100 MB
const float SYNC_WINDOW = 0.8;
const int MAX_RUNNING_JOBS = std::thread::hardware_concurrency() * 2;

std::string COLOR_TRACK_KEY = "COLOR";
std::string DEPTH_TRACK_KEY = "DEPTH";
std::string IR_TRACK_KEY = "INFRARED";

namespace extract_mkv {

  TimesynchronizerPCPD::TimesynchronizerPCPD(ExportConfig &export_config) 
    : TimesynchronizerBase(export_config) {
      spdlog::info("Starting timesynchronized PCPD exported");
      spdlog::info(export_config);
      if (m_export_config.process_color() || m_export_config.process_infrared()) {
        // properly init cuda device
        int gpu_id = pcpd::processing::cuda::initCudaDevice(-1);

        spdlog::info("gpu-id: {0}", gpu_id);

        checkCudaErrors(cudaSetDevice(gpu_id));

        CUdevice cu_device;
        NVDEC_API_CALL(cuCtxGetDevice(&cu_device));
        spdlog::info("cuda device-id: {0}", cu_device);

        checkCudaErrors(cuCtxGetCurrent(&m_cu_context));
      }
  };

  void TimesynchronizerPCPD::initialize_feeds(std::vector<fs::path> input_paths, fs::path output_directory) {
      std::string export_name;
      for(auto input_dir : input_paths) {
          // TODO: this is annoying.. as dir has to end in /. make more robust?
          std::string feed_name = input_dir.parent_path().filename().string();
          export_name = input_dir.parent_path().parent_path().filename().string();
          spdlog::info("Initializing {0}", feed_name);
          // append the appropriate directory onto the output path, i.e. cn01 cn02 cn03..
          auto frame_extractor = std::make_shared<PCPDFileChannel>(input_dir, output_directory / feed_name, feed_name, m_export_config, m_cu_context);
          m_input_feeds.push_back(frame_extractor);
      }

      spdlog::info("Finished initializing feeds, beginning frame export");
      if (m_export_config.export_color_video) {
        m_video_writer = PCPDVideoWriter(input_paths.size(), (output_directory / (export_name + ".avi")).string(), cv::Size(1280, 720));
        if (!m_video_writer.initialize_writer()) {
            spdlog::error("Failed to initialize video writer.");
        };
      }
      int fps_mult = 10;
      float fpns = (1 / m_input_feeds[0]->m_recording_fps) * pow(10, 6);
      m_sync_window = std::chrono::microseconds(static_cast<uint64_t>(SYNC_WINDOW * fpns));
  };

  void TimesynchronizerPCPD::run() {
      // naive counter for successive failures
      m_is_running = true;
      m_thread_pool.push_task([=] () {
        performance_monitor();
      });
      m_thread_pool.push_task([=] () {
          monitor_frame_map();
      });

      while(m_frames_exported < m_export_config.max_frames_exported && m_is_running) {
        spdlog::debug("Exporting frame {0} of frames {1}", m_frames_exported, m_export_config.max_frames_exported);
        for (auto feed : m_input_feeds) {
          advance_feed(feed);
        }
        ++m_frames_exported;
        while (m_export_config.timesync && m_is_running) {
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
            if (feed->m_last_depth_ts == std::chrono::nanoseconds(0)) {
              // TODO: is this still relevant? Do we need other criteria here?
              // ignore feeds for which data has stopped
              continue;
            } else if (diff > m_sync_window) {
              spdlog::warn("Frame: {0} - Feed {1} out of sync at {2}. Feed {3} is ahead at {4}, fast forward..",
                  m_frames_exported, feed->m_feed_name, feed->m_last_depth_ts.count(), first_feed->m_feed_name, first_feed->m_last_depth_ts.count());
              // just advance, basically drop frames for events
              // which we don't have all information present..
              advance_feed(feed);
              sync_cond = false;
            }
          }
          if (sync_cond)
            break;
        }
        std::chrono::nanoseconds mean_timestamp = std::accumulate(m_input_feeds.begin(), m_input_feeds.end(),
              std::chrono::nanoseconds{0},
              [] (std::chrono::nanoseconds acc, auto val) {
                return acc + val->m_last_depth_ts; 
              }
        );
        mean_timestamp = std::chrono::nanoseconds{(mean_timestamp.count()) / m_input_feeds.size()};
        // should not have 4 frames that are in sync..
        if (mean_timestamp < m_export_config.start_ts || mean_timestamp > m_export_config.end_ts) {
          spdlog::warn("Skipping frame, not in indicated time range. {0} not in {1}, {2}",
                       mean_timestamp.count(), m_export_config.start_ts.count(), m_export_config.end_ts.count());
          continue;
        }

        int frame_id = get_frame_from_timestamp(mean_timestamp);

        if (frame_id < 0) {
          spdlog::warn("Timestamp outside of export range.. {0}", m_frames_exported);
          continue;
        }
        // check if frame has appeared before... probably shouldn't
        if (m_frame_map.find(frame_id) != m_frame_map.end()) {
          spdlog::warn("Duplicate frame {0} detected according to timestamping.. skipping", frame_id);
          continue;
        }

        if (m_thread_pool.get_tasks_total() > MAX_RUNNING_JOBS) {
          spdlog::debug("Total tasks qeued: {0}. sleeping", m_thread_pool.get_tasks_total());
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        std::vector<std::shared_future<bool>> futures;

        if (m_export_config.grouped_process()) {
          m_frame_map_lock.lock();
          m_frame_map[frame_id] = std::make_unique<DataGroup>();
          m_frame_map_lock.unlock();
        }

        spdlog::debug("Recieved frame timestamp {0}.. resetting frame count", mean_timestamp.count());
        for (auto feed : m_input_feeds) {
          // reset frame count, we received a frame.
          feed->m_missing_frame_count = 0;

          if (frame_id % m_export_config.skip_frames != 0) {
            continue;
          }

          spdlog::debug("Submitting process feed task for {0}, {1}", feed->m_feed_name, frame_id);

          std::shared_ptr<ProcessedData> processed_data = std::move(feed->m_processed_data);
          if (m_export_config.grouped_process()) {
            m_frame_map_lock.lock();
            m_frame_map[frame_id]->feed_data_map[feed->get_feed_id()] = processed_data;
            m_frame_map_lock.unlock();
          }
          process_feed(feed, processed_data, frame_id);
          /*
          std::shared_future<bool> task_future = m_thread_pool.submit(
               [=] {
                 // TODO: need a way of synchronizing and waiting until we can delete
                 // from frame map
                 processed_data->lock.lock();
                 process_feed(feed, processed_data, frame_id);
                 processed_data->lock.unlock();
               }
          );
          */
        }
        spdlog::debug("Submitting monitor task {0}", frame_id);
      }
      // process remaining frames
      m_is_running = false;
      if (m_export_config.grouped_process()) {
        monitor_frame_map(true);
      }
      spdlog::info("Finishing.. total {0} frames exported", m_frames_exported / m_export_config.skip_frames);
      shutdown();
  };

  void TimesynchronizerPCPD::advance_feed(std::shared_ptr<PCPDFileChannel> feed) {
    using namespace std::chrono;
    // naive counter for debug export
    feed->m_processed_data = std::make_unique<ProcessedData>();
    feed->m_frame_counter = m_frames_exported;
    bool ts_exported = false;
    // create wrappers to get transformation / calibration handles
    microseconds image_timestamp{0};
    // note: lifecycle is not managed by this function. They are stored and cleaned up later.
    feed->m_processed_data->feed_name = feed->m_feed_name;
    if (m_export_config.process_color()) {
      if (feed->pcpd_extract_color(feed->m_processed_data->color_image, image_timestamp, false)) {
        ts_exported = true;
      }
    }
    if (m_export_config.process_depth()) {
      if (feed->pcpd_extract_depth(feed->m_processed_data->depth_image, image_timestamp, false)) {
        // TODO: we want to use actual depth timecode timestamps at some point..
        // need to get these out of the mkvs
        ts_exported = true;
      }
    }

    if (m_export_config.process_infrared()) {
      if (feed->pcpd_extract_infrared(feed->m_processed_data->ir_image, image_timestamp, false)) {
        ts_exported = true;
      }
    }

    if (ts_exported) {
      feed->m_last_depth_ts = image_timestamp;
      feed->m_processed_data->timestamp_us = image_timestamp;
    } else {
      spdlog::error("Color or depth must be exported, or no time information is present!");
      feed->m_missing_frame_count++;
      feed->m_last_depth_ts = microseconds(0);
    }

    if (feed->m_missing_frame_count > 100) {
      // too many missing frames
      spdlog::warn("Too many missing frames from feed {0}, exiting", feed->m_feed_name);
      m_is_running = false;
    }
  };

  bool TimesynchronizerPCPD::process_feed(std::shared_ptr<PCPDFileChannel> feed,
      std::shared_ptr<ProcessedData> data, const int frame_id) {
    // spdlog::trace("Got timestamp {0} from feed {1}", image_timestamp.count(), feed->m_feed_name);

    auto k4a_wrapper = std::make_shared<KPU::Kinect4AzureCaptureWrapper>(feed->m_feed_name);
    assert(k4a_wrapper->capture_handle != nullptr);

    uint64_t depth_ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(data->timestamp_us).count();
    if (m_export_config.process_depth()) {
      // depth needs to be set on wrapper
      uint8 bits_per_element = 16;
      size_t stride = (feed->m_depth_image_width * bits_per_element) / 8;
      size_t nbytes = stride * feed->m_depth_image_height;
      bool ret = k4a_wrapper->setDepthImage(depth_ts_ns,
                                            depth_ts_ns,
                                            feed->m_depth_image_width, feed->m_depth_image_height,
                                            stride, (void*) (data->depth_image.data),
                                            nbytes);
      if (!ret)
        spdlog::warn("Failed to set depth image on wrapper!");

    }

    if (m_export_config.export_depth) {
        process_depth(k4a_wrapper->capture_handle.get_depth_image(),
                      feed->m_device_wrapper,
                      feed->get_output_dir(),
                      frame_id);
    }

    cv::Mat color_image;
    if (m_export_config.process_color()) {
      if (data->color_image.empty()) {
        spdlog::warn("Color image incomplete! {0} for feed {1}", frame_id, data->feed_name);
        return false;
      }
      //process_depth_raw(data->depth_image, std::string("before_download_depth"), feed->get_output_dir(), frame_id);
      data->color_image.download(color_image);
      //process_depth_raw(data->depth_image, std::string("after_download_depth"), feed->get_output_dir(), frame_id);
      // color needs to be set on wrapper
      uint64_t depth_ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(data->timestamp_us).count();
      size_t stride = (data->color_image.cols * data->color_image.elemSize());
      size_t nbytes = stride * data->color_image.rows;
      bool ret = k4a_wrapper->setColorImage(depth_ts_ns,
                                            depth_ts_ns, feed->m_color_image_width,
                                            feed->m_color_image_height, stride,
                                            (void*) (color_image.data),
                                            nbytes, feed->m_color_pixel_format);

      if (!ret)
        spdlog::warn("Failed to set color image on wrapper!");

    }

    if (m_export_config.process_infrared()) {
      cv::Mat image;
      uint8 bits_per_element = 16;
      size_t stride = (feed->m_depth_image_width * bits_per_element) / 8;
      size_t nbytes = stride * feed->m_depth_image_height;
      bool ret = k4a_wrapper->setInfraredImage(depth_ts_ns,
                                               depth_ts_ns, 
                                               feed->m_depth_image_width, feed->m_depth_image_height,
                                               stride, (void*) (data->ir_image.data),
                                               nbytes);
      if (!ret)
        spdlog::warn("Failed to set depth image on wrapper!");

    }

    // spdlog::trace("K4AWrapper pointer count, beginning of advance: {0}", k4a_wrapper.use_count());
    if (m_export_config.export_color) {
        process_color(k4a_wrapper->capture_handle.get_color_image(),
                      feed->m_device_wrapper,
                      feed->get_output_dir(),
                      frame_id);
    }

    if (m_export_config.export_infrared) {
      process_ir(k4a_wrapper->capture_handle.get_ir_image(),
                 feed->m_device_wrapper,
                 feed->get_output_dir(),
                 frame_id);
    }

    if (m_export_config.export_rgbd) {
      feed->m_transformation.process_rgbd(
          k4a_wrapper->capture_handle.get_depth_image(),
          feed->m_color_image_width, feed->m_color_image_height,
          feed->m_device_wrapper,
          feed->get_output_dir(),
          frame_id
      );
    }

    if (m_export_config.export_pointcloud) {
      feed->m_transformation.process_pointcloud(
          k4a_wrapper->capture_handle.get_color_image(),
          k4a_wrapper->capture_handle.get_depth_image(),
          feed->m_device_wrapper,
          feed->get_output_dir(),
          frame_id);
    }

    if (m_export_config.export_bodypose) {
      process_pose(feed->m_device_wrapper, feed->get_output_dir(), frame_id);
    }

    // spdlog::trace("K4AWrapper pointer count, end of advance: {0}", k4a_wrapper.use_count());
    return true;
  };

  void TimesynchronizerPCPD::monitor_frame_map(bool flush /* = false */) {
    // map should be ordered, so we can process in this manner
    // TODO: make this buffer size configurable
    spdlog::debug("Starting monitor.. flush={0}", flush);
    while (m_is_running) {
      int MAX_FRAME_BUFFER_SIZE = 10;
      int map_size = m_frame_map.size();
      if (map_size < MAX_FRAME_BUFFER_SIZE)
        // allow the feeds some time to catch up..
        continue;
      auto element = m_frame_map.cbegin();
      // always leave a buffer window of 30
      // need to independently monitor frames so they are all more or less in sync;
      int i = 0;
      while (element != m_frame_map.cend() && (map_size - i > MAX_FRAME_BUFFER_SIZE || flush)) {
        // debug printing
        if (element->second->feed_data_map.size() > 0) {
          spdlog::trace("Frame map for frame {0}", element->first);
          for (const auto& it : element->second->feed_data_map) {
            spdlog::trace(it.second->feed_name);
          }
        }
        std::chrono::microseconds timestamp;
        if (m_export_config.export_color_video && element->second->feed_data_map.size() == m_input_feeds.size()) {
          std::map<int, cv::cuda::GpuMat> color_images;
          for (auto it : element->second->feed_data_map) {
            auto wrapper = it.second;
            // lock each wrapper..
            wrapper->lock.lock();
            if (!wrapper->color_image.empty()) {
              int feed_id = it.first;
              color_images[feed_id] = wrapper->color_image;
              timestamp = wrapper->timestamp_us;
            };
          }
          //all images present
          m_video_writer.write_frames(color_images, timestamp);
          for (auto it : element->second->feed_data_map) {
            it.second->lock.unlock();
          }
          // standard associative-container erase idiom.
        }
        spdlog::debug("Removing iterated map element");

        m_frame_map_lock.lock();
        // TODO: need to make sure individual items are done processing
        m_frame_map.erase(element++);
        m_frame_map_lock.unlock();
        i++;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    spdlog::debug("Exiting monitor, running: {0}", m_is_running);
  };

  PCPDFileChannel::PCPDFileChannel(fs::path input_dir, fs::path output_directory,
                                   std::string feed_name, ExportConfig &export_config, CUcontext &cu_context) :
                                  m_output_dir(output_directory), m_feed_name(feed_name),
                                  m_export_config(export_config) {
    // sets are sorted automatically
    std::set<std::string> filepaths;
    for (const auto & entry : fs::directory_iterator(input_dir)) {
        auto ext = entry.path().extension().string();
        if (ext.find("mkv") != std::string::npos && fs::file_size(entry) > MIN_FILESIZE_BYTES) {
          filepaths.insert(entry.path().string());
        } else {
          spdlog::warn("Filepath requirements not met for {0}", entry.path().string());
        }
    }

    fs::create_directories(m_output_dir);

    std::vector<std::string> fps{filepaths.begin(), filepaths.end()};
    if (fps.empty()) {
      spdlog::warn("No filepaths found for feed {0}", m_feed_name);
    } else {
      for (auto &fp : fps) {
        spdlog::info(fp);
      }
    }
    double ns_per_frame = std::pow(10, 9) / static_cast<double>(m_recording_fps);

    spdlog::info("Setting start time {0} and end time {1} for channel {2}", export_config.start_ts.count(), export_config.end_ts.count(), m_feed_name);
    // Subtract offset for decoder.. bgroup size should be 5. otherwise images come out with artifacts.
    MkvTrackLoaderConfig trackloader_config{};
    trackloader_config.start_timestamp_offset_ns = export_config.start_ts.count() - 5 * ns_per_frame;
    trackloader_config.end_timestamp_offset_ns = export_config.end_ts.count();
    trackloader_config.file_paths = fps;
    trackloader_config.nth_frame = 1;
    // hack alert... insert fake locator
    std::shared_ptr<service::Locator> spLocator = nullptr;
    m_loader = std::make_shared<MkvSeekTrackLoader>(spLocator, trackloader_config);
    // TODO: need to get height/width of images.
    m_color_decoder = std::make_shared<H264Decoder>(cu_context, m_color_image_width, m_color_image_height,
                                                    m_feed_name, DECODER_TYPE::COLOR);
    m_ir_decoder = std::make_shared<H264Decoder>(cu_context, m_depth_image_width, m_depth_image_height,
                                                 m_feed_name, DECODER_TYPE::IR);
    if (m_export_config.process_color())
      m_loader->addTrack(COLOR_TRACK_KEY);
    if (m_export_config.process_depth())
      m_loader->addTrack(DEPTH_TRACK_KEY);
    if (m_export_config.process_infrared())
      m_loader->addTrack(IR_TRACK_KEY);
    // read info from first filepath
    load_mkv_info(fps[0]);
    // need to call this last.. once we have calibration, etc
    // NOTE: recording fps not configurable right now
    initialize();
    //print_raw_calibration(m_calibration);
    m_transformation.init_transformation(m_calibration);
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
    Eigen::Matrix4f color2depth, color2depth_opencv;
    device_calibration.color2depth_transform.toMatrix4f(color2depth);
    opengl_to_opencv_transform(color2depth, color2depth_opencv);

    device_calibration.color2depth_transform.rotation = color2depth_opencv.block<3, 3>(0, 0);
    device_calibration.color2depth_transform.translation = color2depth_opencv.block<3, 1>(0, 3);
    KPU::deserializeCalibrations(reader, device_calibration);
    KPU::toK4A(device_calibration, m_calibration);
    spdlog::info("Got extrinsics.. {0}", device_calibration.camera_pose);
    Eigen::Matrix4f extrinsics_gl;
    device_calibration.camera_pose.toMatrix4f(extrinsics_gl);
    write_rigid_transform(device_calibration.camera_pose);
    opengl_to_opencv_transform(extrinsics_gl, m_extrinsics);
  }

  void PCPDFileChannel::write_rigid_transform(pcpd::datatypes::RigidTransform& transform) {
    // write camera extrinsics
    std::ofstream file_id;
    fs::path filename = fs::path(m_output_dir) / "world2camera.json";
    Json::StreamWriterBuilder wbuilder;
    wbuilder["indentation"] = "\t";
    Json::Value root;   // will contains the root value after parsing.
    root["rotation"]["w"] = transform.rotation.w();
    root["rotation"]["x"] = transform.rotation.x();
    root["rotation"]["y"] = transform.rotation.y();
    root["rotation"]["z"] = transform.rotation.z();
    root["translation"]["m00"] = transform.translation[0];
    root["translation"]["m10"] = transform.translation[1];
    root["translation"]["m20"] = transform.translation[2];
    std::string document = Json::writeString(wbuilder, root);
    file_id.open(filename.c_str(), std::ios::out);
    file_id << document << std::endl;
    file_id.close();
  }

  void PCPDFileChannel::write_rigid_transform(Eigen::Matrix4f& transform) {
    Eigen::Matrix3f rot_mat = transform.block<3,3>(0,0);
    Eigen::Quaternionf rot(rot_mat);
    Eigen::Vector3f trans = transform.block<1, 3>(0, 0);
    std::ofstream file_id;
    fs::path filename = fs::path(m_output_dir) / "world2camera.json";
    Json::StreamWriterBuilder wbuilder;
    wbuilder["indentation"] = "\t";
    Json::Value root;   // will contains the root value after parsing.
    root["rotation"]["w"] = rot.w();
    root["rotation"]["x"] = rot.x();
    root["rotation"]["y"] = rot.y();
    root["rotation"]["z"] = rot.z();
    root["translation"]["m00"] = trans[0];
    root["translation"]["m10"] = trans[1];
    root["translation"]["m20"] = trans[2];
    std::string document = Json::writeString(wbuilder, root);
    file_id.open(filename.c_str(), std::ios::out);
    file_id << document << std::endl;
    file_id.close();
  }

  void PCPDFileChannel::initialize() {
    spdlog::info("Initializing device calibration");
    m_device_wrapper = std::make_shared<K4ADeviceWrapper>();
    m_device_wrapper->calibration = m_calibration;
    RectifyMaps rectify_maps = process_calibration(m_calibration, m_output_dir);
    m_device_wrapper->rectify_maps = rectify_maps;
    spdlog::info("Done initializing device calibration");
  }

  int64_t TimesynchronizerPCPD::get_frame_from_timestamp(std::chrono::nanoseconds timestamp) {
    // timestamp should be in nanoseconds
    double ns_per_frame = std::pow(10, 9) / static_cast<double>(m_recording_fps);
    return std::round(static_cast<double>((timestamp - m_export_config.start_ts).count()) / ns_per_frame);
  }

  bool PCPDFileChannel::pcpd_extract_color(cv::cuda::GpuMat &color_image, std::chrono::microseconds &timestamp, bool write) { 
    MkvDataBlock2 block;
    if(!m_loader->getNextDataBlock(block, COLOR_TRACK_KEY)) {
      spdlog::error("Unable to get next color block {0}..", m_frame_counter);
      return false;
    }

    spdlog::trace("Extracting color image from mkv");
    color_image.create(cv::Size(m_color_image_width, m_color_image_height), CV_8UC4);
    bool ret = m_color_decoder->decode(block.data->data_block, color_image);
    timestamp = std::chrono::microseconds{block.data->device_timestamp_usec};
    if (!ret) {
      spdlog::error("Export failed for frame {0}", m_frame_counter);
    } else {
      if (write) {
        auto path_template = "color_frame_{0}.jpg";
        spdlog::info("Export color frame: {0} with size {1} and timestamp: {2}", m_frame_counter, color_image.size(), block.data->device_timestamp_ns);
        fs::path fp = (m_output_dir / fmt::format(path_template, m_frame_counter)).string();
        cv::Mat out;
        color_image.download(out);
        cv::imwrite(fp, out);
      }
    }
    return true;
  }

  bool PCPDFileChannel::pcpd_extract_depth(cv::Mat& depth_image, std::chrono::microseconds &timestamp, bool write /* =false */) {
    auto zdepth_compressor = std::make_shared<zdepth::DepthCompressor>();
    uint8 bits_per_element = 16;
    MkvDataBlock2 block;
    if(!m_loader->getNextDataBlock(block, DEPTH_TRACK_KEY)) {
      spdlog::error("Unable to get next depth block {0}..", m_frame_counter);
      return false;
    }
    int width;
    int height;

    spdlog::debug("Extracting depth image from mkv");
    size_t n_bytes = m_depth_image_height * m_depth_image_width * bits_per_element;
    depth_image.create(cv::Size(m_depth_image_width, m_depth_image_height), CV_16UC1);
    std::vector<uint16_t> depth_wrapper {};
    wrapArrayInVector((uint16_t *)depth_image.data, n_bytes, depth_wrapper);
    zdepth::DepthResult ret = zdepth_compressor->Decompress(block.data->data_block, width, height, depth_wrapper);
    releaseVectorWrapper(depth_wrapper);
    timestamp = std::chrono::microseconds{block.data->device_timestamp_usec};

    assert(width == m_depth_image_width);
    assert(height == m_depth_image_height);

    // depth_image = cv::Mat(height, width, CV_16U, depth_out.data());
    // TODO: how to make depth_out thread safe?
    // can we decompress directly into depth_image
    //cv::Mat tmp_data{height, width, CV_16U, depth_out.data()};
    //tmp_data.copyTo(depth_image);

    if (ret != zdepth::DepthResult::Success) {
      spdlog::error("Export failed for frame {0}", m_frame_counter);
      return false;
    }
    if (write) {
      auto path_template = "depth_frame_{0}.tiff";
      spdlog::info("Export depth frame: {0} with size {1} and timestamp: {2}",
                   m_frame_counter, depth_image.size(), block.data->device_timestamp_ns);
      fs::path fp = (m_output_dir / fmt::format(path_template, m_frame_counter)).string();
      cv::Mat image;
      depth_image.convertTo(image, CV_16UC1);
      cv::imwrite(fp, image);
    }
    return true;
  }

  bool PCPDFileChannel::pcpd_extract_infrared(cv::Mat& infrared_image, std::chrono::microseconds& timestamp, bool write) {
    uint8 bits_per_element = 16;
    MkvDataBlock2 block;
    if(!m_loader->getNextDataBlock(block, IR_TRACK_KEY)) {
      spdlog::error("Unable to get next infrared block {0}..", m_frame_counter);
      return false;
    }

    timestamp = std::chrono::microseconds{block.data->device_timestamp_usec};
    spdlog::debug("Extracting infrared image from mkv");
    bool ret = m_ir_decoder->decode(block.data->data_block, infrared_image);
    if (!ret) {
      spdlog::error("Export failed for frame {0}", m_frame_counter);
    } else {
      if (write) {
        auto path_template = "infrared_frame_{0}.tiff";
        spdlog::trace("Export infrared frame: {0} with size {1} and timestamp: {2}", m_frame_counter, infrared_image.size(), block.data->device_timestamp_ns);
        fs::path fp = (m_output_dir / fmt::format(path_template, m_frame_counter)).string();
        cv::Mat image;
        infrared_image.convertTo(image, CV_8UC1);
        cv::imwrite(fp, image);
      }
    }
    return true;
  }

  std::string PCPDFileChannel::get_output_dir() {
    return m_output_dir;
  }
}
