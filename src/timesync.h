#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>

#include <extract_mkv_k4a.h>


namespace extract_mkv {

  const int MAX_PARALLEL_JOBS = 12;

  class Semaphore {
  public:
      Semaphore (int count_ = 0)
          : count(count_) {}

      inline void notify()
      {
          std::unique_lock<std::mutex> lock(mtx);
          count++;
          cv.notify_one();
      }

      inline void wait()
      {
          std::unique_lock<std::mutex> lock(mtx);

          while(count == 0){
              cv.wait(lock);
          }
          count--;
      }

  private:
      std::mutex mtx;
      std::condition_variable cv;
      int count;
  };

  class Timesynchronizer { 
public: explicit Timesynchronizer(bool, bool, bool, bool, bool, bool, size_t, size_t);
      void initialize_feeds(std::vector<fs::path>, fs::path);
      void feed_forward(int);
      void extract_frames(std::shared_ptr<K4AFrameExtractor>, int);
      void run();
      void remove_thread(std::thread::id);

    protected:
      std::vector<std::shared_ptr<K4AFrameExtractor>> m_input_feeds;
      const bool m_export_timestamp{false};
      const bool m_export_color{false};
      const bool m_export_depth{false};
      const bool m_export_infrared{false};
      const bool m_export_pointcloud{false};
      const bool m_export_rgbd{false};

      const size_t m_first_frame;
      const size_t m_last_frame;
      float m_sync_window;

      std::mutex m_lock;
      Semaphore m_sem{MAX_PARALLEL_JOBS};
      std::vector<std::thread> m_worker_threads;
  };

}
