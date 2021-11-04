#ifdef __GNUC__
#if __GNUC__ >= 9
#include <execution>
#include <tbb/tbb.h>
#endif
#endif
#include <spdlog/spdlog.h>
#include <future>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "extract_mkv/timesync.h"
#include "extract_mkv/extract_mkv_k4a.h"

using namespace extract_mkv;

namespace extract_mkv {
  Timesynchronizer::Timesynchronizer(const size_t first_frame, const size_t last_frame,
      const size_t skip_frames, ExportConfig export_config, const bool timesync, const bool enable_seek) :
    m_export_config(export_config), m_first_frame(first_frame), m_last_frame(last_frame), 
    m_skip_frames(skip_frames), m_use_timesync(timesync), m_enable_seek(enable_seek) {
      spdlog::info("Initialized Timesynchronizer. timesync: {0}", timesync);
    };

  void Timesynchronizer::initialize_feeds(std::vector<fs::path> input_paths, fs::path output_directory) {
#if __GNUC__ >= 9
      std::for_each(std::execution::par, input_paths.begin(), input_paths.end(),
          [=, this](auto&& input_dir) {
          std::string feed_name = input_dir.parent_path().filename().string();
          spdlog::info("Initializing {0}", feed_name);
          // append the appropriate directory onto the output path, i.e. cn01 cn02 cn03..
          auto frame_extractor = std::make_shared<K4AFrameExtractor>(input_dir, output_directory / feed_name, feed_name, m_export_config);
          std::scoped_lock<std::mutex> guard(m_lock);
          m_input_feeds.push_back(frame_extractor);
      });
#else
      // TODO: could do easy parallel join threads for gcc 7
      for(auto input_dir : input_paths) {
          std::string feed_name = input_dir.parent_path().filename().string();
          spdlog::info("Initializing {0}", feed_name);
          // append the appropriate directory onto the output path, i.e. cn01 cn02 cn03..
          auto frame_extractor = std::make_shared<K4AFrameExtractor>(input_dir, output_directory / feed_name, feed_name, m_export_config);
          m_input_feeds.push_back(frame_extractor);
      }
#endif
      auto base_fps = m_input_feeds[0]->get_fps();
      if (!std::all_of(m_input_feeds.begin() + 1, m_input_feeds.end(), 
            [&] (auto feed) {return feed->get_fps() == base_fps;}) ) {
          spdlog::error("Not all fps of recordings are equal!");
          throw 1;
      }
      // sync_window in microseconds
      float fps = 0.8 * (1 / static_cast<float>(base_fps)) * pow(10, 6);
      m_sync_window = std::chrono::microseconds(static_cast<uint64_t>(fps));
  };

  void Timesynchronizer::feed_forward(int frame_counter) {
      for (auto feed : m_input_feeds) {
          feed->next_capture();
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
          auto diff = first_feed->m_last_depth_ts - feed->m_last_depth_ts;
          spdlog::trace("Feed {0} syncing diff: {1} - syncwindow: {2}", feed->m_name, diff.count(), m_sync_window.count());
          if (diff > m_sync_window) {
            spdlog::warn("Frame: {0} - Feed {1} out of sync at {2}. Feed {3} is ahead at {4}, fast forward..",
                frame_counter, feed->m_name, feed->m_last_depth_ts.count(), first_feed->m_name, first_feed->m_last_depth_ts.count());
            break_cond = false;
            feed->next_capture();
          }
        }
        if (break_cond)
          break;
      }
  };

  void Timesynchronizer::monitor() {
    while (m_is_running) {
      spdlog::debug("Cleaning up threads.."); 
      std::unique_lock<std::mutex> lock1(m_thread_free_lock);
      m_wait_cv.wait(lock1);

      std::scoped_lock<std::mutex> lock2(m_lock);
      auto iter = m_finished_threads.begin();
      while (iter != m_finished_threads.end()) {

        auto id = *iter;

        auto found = std::find_if(m_worker_threads.begin(), m_worker_threads.end(),
            [=](std::thread &t) { return (t.get_id() == id); });
        if (found != m_worker_threads.end())
        {
            found->join();
            m_worker_threads.erase(found);
        }
        iter = m_finished_threads.erase(iter);
      }
    }
  }

  void Timesynchronizer::run() {

      m_is_running = true;
      m_monitor_thread = std::thread([=] () {
        monitor();
      });
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

              /*
              for (auto feed : m_input_feeds) {
                feed->extract_frames(frame_counter);
              }
              */
              for (auto feed : m_input_feeds) {
                m_sem.wait();
                std::scoped_lock<std::mutex> lock1(m_lock);
                m_worker_threads.push_back(std::thread([=]
                  (std::shared_ptr<K4AFrameExtractor> feed, const int frame_counter) {
                    spdlog::debug("Initializing worker thread for {0} - {1}", feed->m_name, frame_counter); 
                    feed->extract_frames(frame_counter);
                    std::scoped_lock<std::mutex> lock2(m_thread_free_lock);
                    m_finished_threads.push_back(std::this_thread::get_id());
                    m_sem.notify();
                    m_wait_cv.notify_one();
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
      m_is_running = false;
      m_wait_cv.notify_all();
      spdlog::trace("Waiting for monitor...");
      m_monitor_thread.join();
      spdlog::trace("Monitor done...");
      // wait for worker threads to complete

      std::scoped_lock<std::mutex> lock(m_lock);
      for (auto &thread : m_worker_threads) {
        if (thread.joinable()) {
          auto id = std::hash<std::thread::id>{}(thread.get_id());
          spdlog::trace("Waiting for thread.. {0}", id);
          thread.join();
          spdlog::trace("Thread Done.. {0}", id);
        }
      }
  }
}
