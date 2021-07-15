#include <execution>
#include <tbb/tbb.h>
#include <spdlog/spdlog.h>
#include <future>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <timesync.h>
#include <extract_mkv_k4a.h>

using namespace extract_mkv;

namespace extract_mkv {
  Timesynchronizer::Timesynchronizer(size_t first_frame, size_t last_frame,
      size_t skip_frames, ExportConfig export_config, bool timesync, bool enable_seek) :
    m_export_config(export_config), m_first_frame(first_frame), m_last_frame(last_frame), 
    m_skip_frames(skip_frames), m_use_timesync(timesync), m_enable_seek(enable_seek) {
      spdlog::info("Initialized Timesynchronizer. timesync: {0}", timesync);
    };

  void Timesynchronizer::initialize_feeds(std::vector<fs::path> input_paths, fs::path output_directory) {
      std::for_each(std::execution::par, input_paths.begin(), input_paths.end(),
          [=, this](auto&& input_dir) {
          std::string feed_name = input_dir.parent_path().filename().string();
          spdlog::info("Initializing {0}", feed_name);
          // append the appropriate directory onto the output path, i.e. cn01 cn02 cn03..
          auto frame_extractor = std::make_shared<K4AFrameExtractor>(input_dir, output_directory / feed_name, feed_name, m_export_config);
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
          feed->next_capture(frame_counter);
      }
      if (!m_use_timesync) {
        return;
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
        spdlog::trace("Syncing feeds");
        for (auto feed : m_input_feeds) {
          // TODO: sync both color and depth? or just color
          if ((first_feed->m_last_depth_ts - feed->m_last_depth_ts) > m_sync_window ||
               (first_feed->m_last_color_ts - feed->m_last_color_ts) > m_sync_window) {
            spdlog::warn("Frame: {0} - Feed {1} out of sync at {2}. Feed {3} is ahead at {4}, fast forward..",
                frame_counter, feed->m_name, feed->m_last_depth_ts, first_feed->m_name, first_feed->m_last_depth_ts);
            break_cond = false;
            feed->next_capture(frame_counter);
          }
        }
        if (break_cond)
          break;
      }
  };

  void Timesynchronizer::run() {

      // now export frames
      int frame_counter{0};
      int wait_count{0};
      std::thread t;
      // DOES NOT WORK FOR SOME RECORDINGS!
      if (m_enable_seek && m_first_frame > 0) {
        for (auto feed : m_input_feeds) {
          feed->seek(m_first_frame);
        }
        frame_counter = m_first_frame;
      }

      while (!m_last_frame || frame_counter <= m_last_frame) {
          try {

              // feed forward happens in sync.. processing does not.
              feed_forward(frame_counter);

              if (frame_counter < m_first_frame || frame_counter % m_skip_frames != 0) {
                  frame_counter++;
                  spdlog::trace("Skipping frame: {0}", frame_counter);
                  continue;
              }
              spdlog::debug("Extract Frame: {0}", frame_counter);

              for (auto feed : m_input_feeds) {
                m_sem.wait();
                m_worker_threads.push_back(std::thread([=, this]
                  (std::shared_ptr<K4AFrameExtractor> feed, const int frame_counter) {
                    spdlog::debug("Initializing worker thread for {0} - {1}", feed->m_name, frame_counter); 
                    feed->extract_frames(frame_counter);
                    m_sem.notify();
                    // TODO: this is ugly way to clean up threads, maybe we want to
                    // use an async monitor based approach.. and check if they are done.
                    std::thread([=, this] (std::thread::id thread_id) {
                        this->remove_thread(thread_id);
                    }, std::this_thread::get_id()).detach();
                }, feed, frame_counter));
              }

              ++frame_counter;

          } catch (k4a::error &e) {
              if (std::string(e.what()) == "Failed to get next capture!") {
                  spdlog::error("{0}, continuing..", e.what());
                  continue;
              } else {
                  spdlog::error("Error during playback {0}.", e.what());
                  exit(1);
              }
          }
      }
  }

  void Timesynchronizer::remove_thread(std::thread::id id) {
      spdlog::debug("Clean up thread"); 
      std::scoped_lock<std::mutex> lock(m_lock);
      auto iter = std::find_if(m_worker_threads.begin(), m_worker_threads.end(),
          [=](std::thread &t) { return (t.get_id() == id); });
      if (iter != m_worker_threads.end())
      {
          iter->join();
          m_worker_threads.erase(iter);
      }
  }
}
