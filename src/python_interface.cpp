#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

#include <extract_mkv/timesync.h>
#include <extract_mkv/extract_mkv_k4a.h>
#include "extract_mkv/filesystem.h"


#ifdef WITH_PCPD
// not sure why ringbuffer is not linked properly
#include <extract_mkv/pcpd_file_exporter.h>
#endif


namespace py = pybind11;

namespace extract_mkv {

  void set_loglevel(std::string loglevel) {
    if (loglevel == "debug")
        spdlog::set_level(spdlog::level::debug);
    else if (loglevel == "warn")
        spdlog::set_level(spdlog::level::warn);
    else if (loglevel == "trace")
        spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[%H:%M:%S] %^[%l]%$ %l [thread %t] %v");
  }

  PYBIND11_MODULE(mkv_extractor, m) {
    m.def("set_log_level", &set_loglevel);


    py::class_<ExportConfig>(m, "ExportConfig")
      .def(py::init<>())
      .def_readwrite("export_timestamp", &ExportConfig::export_timestamp)
      .def_readwrite("export_color", &ExportConfig::export_color)
      .def_readwrite("export_distorted", &ExportConfig::export_distorted)
      .def_readwrite("export_depth", &ExportConfig::export_depth)
      .def_readwrite("export_infrared", &ExportConfig::export_infrared)
      .def_readwrite("export_rgbd", &ExportConfig::export_rgbd)
      .def_readwrite("export_pointcloud", &ExportConfig::export_pointcloud)
      .def_readwrite("export_color_video", &ExportConfig::export_color_video)
      .def_readwrite("align_clouds", &ExportConfig::align_clouds)
      .def_readwrite("timesync", &ExportConfig::timesync)
      .def_readwrite("export_extrinsics", &ExportConfig::export_extrinsics)
      .def_readwrite("max_frames_exported", &ExportConfig::max_frames_exported)
      .def_readwrite("skip_frames", &ExportConfig::skip_frames)
      .def_readwrite("start_ts", &ExportConfig::start_ts)
      .def_readwrite("end_ts", &ExportConfig::end_ts);

    py::class_<TimesynchronizerK4A>(m, "TimesynchronizerK4A")
        .def(py::init<const size_t, const size_t,
             const size_t, ExportConfig&, const bool,
             const bool>())
        .def("initialize_feeds", &TimesynchronizerK4A::initialize_feeds)
        .def("run", &TimesynchronizerK4A::run);
    py::class_<fs::path>(m, "Path")
        .def(py::init<std::string>());
    py::implicitly_convertible<std::string, fs::path>();
    // py::implicitly_convertible<uint64_t, std::chrono::nanoseconds>();

#ifdef WITH_PCPD
    py::class_<TimesynchronizerPCPD>(m, "TimesynchronizerPCPD")
        .def(py::init<ExportConfig&>())
        .def("initialize_feeds", &TimesynchronizerPCPD::initialize_feeds)
        .def("run", &TimesynchronizerPCPD::run);
#endif
  }
}
