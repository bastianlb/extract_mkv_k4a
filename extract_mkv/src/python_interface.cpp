#pragma once

#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <timesync.h>
#include <extract_mkv_k4a.h>

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
      .def_readwrite("export_depth", &ExportConfig::export_depth)
      .def_readwrite("export_infrared", &ExportConfig::export_infrared)
      .def_readwrite("export_rgbd", &ExportConfig::export_rgbd)
      .def_readwrite("export_pointcloud", &ExportConfig::export_pointcloud)
      .def_readwrite("align_clouds", &ExportConfig::align_clouds)
      .def_readwrite("export_extrinsics", &ExportConfig::export_extrinsics);


    py::class_<Timesynchronizer>(m, "Timesynchronizer")
        .def(py::init<const size_t, const size_t,
             const size_t, ExportConfig, const bool,
             const bool>())
        .def("initialize_feeds", &Timesynchronizer::initialize_feeds)
        .def("run", &Timesynchronizer::run);
    py::class_<std::filesystem::path>(m, "Path")
        .def(py::init<std::string>());
    py::implicitly_convertible<std::string, std::filesystem::path>();
  }
}
