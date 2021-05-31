#include <execution>
#include <tbb/tbb.h>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <timesync.h>
#include <extract_mkv_k4a.h>

u_int8_t MAX_PARALLEL_JOBS = 8;

using namespace extract_mkv;

namespace extract_mkv {
  Timesynchronizer::Timesynchronizer(bool export_color, bool export_depth, bool export_infrared,
                                     bool export_rgbd, bool export_pointcloud, bool export_timestamp,
                                     size_t first_frame, size_t last_frame) :
    m_export_color(export_color), m_export_depth(export_depth), m_export_infrared(export_infrared),
    m_export_rgbd(export_rgbd), m_export_pointcloud(export_pointcloud), m_export_timestamp(export_timestamp),
    m_first_frame(first_frame), m_last_frame(last_frame) {};

  void Timesynchronizer::initialize_feeds(std::vector<fs::path> input_paths, fs::path output_directory) {
      std::for_each(std::execution::par, input_paths.begin(), input_paths.end(),
          [=](auto&& input_dir) {
          std::string feed_name = input_dir.parent_path().filename().string();
          spdlog::info("Initializing {0}", feed_name);
          // append the appropriate directory onto the output path, i.e. cn01 cn02 cn03..
          auto frame_extractor = std::make_shared<K4AFrameExtractor>(input_dir, output_directory / feed_name, feed_name);
          std::scoped_lock<std::mutex> guard(m_lock);
          m_input_feeds.push_back(frame_extractor);
      });
      auto base_fps = m_input_feeds[0]->get_fps();
      if (!std::all_of(m_input_feeds.begin() + 1, m_input_feeds.end(), 
            [&] (auto feed) {return feed->get_fps() == base_fps;}) ) {
          spdlog::error("Not all fps of recordings are equal!");
          throw 1;
      }
      m_sync_window =  (1 / static_cast<float>(base_fps)) * std::pow(10, 9);
  };

  void Timesynchronizer::feed_forward(int frame_counter) {
      for (auto feed : m_input_feeds) {
          feed->next_capture();
      }
      while (true) {
        // fast forward until streams are in sync again
        // look at the feed that is furthest ahead, and sync others to that
        // timepoint if they are lagging behind.
        bool break_cond{true};
        auto first_feed = *std::max_element(m_input_feeds.begin(), m_input_feeds.end(),
              [] (auto lhs, auto rhs) {
              return lhs->m_last_depth_ts < rhs->m_last_depth_ts;
        });
        for (auto feed : m_input_feeds) {
          // TODO: sync both color and depth? or just color
          if ((first_feed->m_last_depth_ts - feed->m_last_depth_ts) > m_sync_window ||
               (first_feed->m_last_color_ts - feed->m_last_color_ts) > m_sync_window) {
            spdlog::warn("Frame: {0} - Feed {1} out of sync at {2}. Feed {3} is ahead at {4}, fast forward..",
                frame_counter, feed->m_name, feed->m_last_depth_ts, first_feed->m_name, first_feed->m_last_depth_ts);
            break_cond = false;
            feed->next_capture();
          }
        }
        if (break_cond)
          break;
      }
  };

  void Timesynchronizer::extract_frames(int frame_counter) {
      try {
          // TODO: only parallelize when rgbd is exported
          std::for_each(std::execution::par, m_input_feeds.begin(), m_input_feeds.end(),
            [=](auto&& feed) {
            // TODO: c++ counting semaphore to enable thread limit?
            // kind of ugly to use mutex on k4a class..
            std::scoped_lock lk(feed->lock);
            spdlog::info("Processing {0} : {1}", feed->m_name, frame_counter);
            if (m_export_depth) {
                feed->process_depth(frame_counter);
            }

            if (m_export_color) {
                feed->process_color(frame_counter);
            }

            if (m_export_infrared) {
                feed->process_ir(frame_counter);
            }

            if (m_export_rgbd) {
                feed->process_rgbd(frame_counter);
            }

            if (m_export_pointcloud) {
                feed->process_pointcloud(frame_counter);
            }
          });

      } catch (const extract_mkv::MissingDataException& e) {
          spdlog::error("Error during playback: {0}", e.what());
      }

      /*tsss << "\n";
      if (m_export_timestamp) {
          Corrade::Utility::Directory::appendString(timestamp_path, tsss.str());
      }*/

  }

  void Timesynchronizer::run() {

      // now export frames
      int frame_counter{0};
      /* TODO: write only relevant headers.
       * if (m_capture_timestamps) {
        m_tsss << "frameindex,depth_dts,depth_sts,color_dts,color_sts,infrared_dts,infrared_sts\n";
      }*/

      while (true) {
          try {
              feed_forward(frame_counter);

              if (frame_counter < m_first_frame) {
                  frame_counter++;
                  continue;
              }

              if (m_last_frame > 0 && frame_counter > m_last_frame) {
                  break;
              }
              spdlog::debug("Extract Frame: {0}", frame_counter);

              extract_frames(frame_counter);

              ++frame_counter;

          } catch (k4a::error &e) {
              if (std::string(e.what()) == "Failed to get next capture!") {
                  spdlog::debug("Playback Stopped.");
                  break;
              } else {
                  spdlog::error("Error during playback {0}.", e.what());
                  exit(1);
              }
          }
      }
  }
}
