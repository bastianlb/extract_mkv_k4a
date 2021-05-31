#pragma once

#include <mutex>

#include <extract_mkv_k4a.h>

namespace extract_mkv {
  class Timesynchronizer { 

    public:
      explicit Timesynchronizer(bool, bool, bool, bool, bool, bool, size_t, size_t);
      void initialize_feeds(std::vector<fs::path>, fs::path);
      void feed_forward(int);
      void extract_frames(int);
      void run();

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
  };
}
