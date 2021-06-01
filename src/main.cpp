#include <fstream>              // File IO
#include <iostream>             // Terminal IO
#include <sstream>              // Stringstreams
#include <exception>
#include <algorithm>
#include <filesystem>

#include <Corrade/configure.h>
#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Directory.h>
#include <Corrade/Containers/ArrayView.h>

#ifdef CORRADE_TARGET_UNIX
#ifdef CORRADE_TARGET_APPLE
#include <Magnum/Platform/WindowlessCglApplication.h>
#else

#include <Magnum/Platform/WindowlessGlxApplication.h>

#endif
#else
#ifdef CORRADE_TARGET_WINDOWS
#include <Magnum/Platform/WindowlessWglApplication.h>
#endif
#endif

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/ArrayView.h>

#include "Magnum/Math/Matrix3.h"
#include "Magnum/Math/Matrix4.h"
#include "Magnum/Math/Range.h"

#include "Magnum/GL/Context.h"

#include "yaml-cpp/yaml.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/daily_file_sink.h>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

#include <timesync.h>

namespace fs = std::filesystem;

namespace Magnum {

    class ExtractFramesMKV : Platform::WindowlessApplication {
    public:
        explicit ExtractFramesMKV(const Arguments &arguments);

        int exec() override;

    private:
        Corrade::Utility::Arguments args;

        size_t m_first_frame{0};
        size_t m_last_frame{0};

        bool m_export_timestamp{false};
        bool m_export_color{false};
        bool m_export_depth{false};
        bool m_export_infrared{false};
        bool m_export_pointcloud{false};
        bool m_export_rgbd{false};

        fs::path m_input_directory;
        fs::path m_output_directory;
        std::vector<fs::path> m_input_feeds;
    };

    ExtractFramesMKV::ExtractFramesMKV(const Arguments &arguments) : Magnum::Platform::WindowlessApplication{
            arguments} {
        args.addOption('c', "config").setHelp("config", "config file for settings")
            .addOption('l', "loglevel", "info").setHelp("loglevel", "spdlog logging")
             .addSkippedPrefix("magnum", "engine-specific options");

        args.parse(arguments.argc, arguments.argv);

        std::string loglevel = args.value<std::string>("loglevel");
        if (loglevel == "debug")
            spdlog::set_level(spdlog::level::debug);
        else if (loglevel == "warn")
            spdlog::set_level(spdlog::level::warn);
        spdlog::set_pattern("[%H:%M:%S] %^[%l]%$ %l [thread %t] %v");
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_st>());
        sinks.push_back(std::make_shared<spdlog::sinks::daily_file_sink_st>("logfile", 23, 59));
        auto combined_logger = std::make_shared<spdlog::logger>("name", begin(sinks), end(sinks));
        //register it if you need to access it globally
        spdlog::register_logger(combined_logger);

        std::string config_path = args.value<std::string>("config");

        YAML::Node config = YAML::LoadFile(config_path);

        if (!config["record"]) {
            spdlog::error("Incorrectly formatted config, exiting."); 
            exit(1);
        }
        YAML::Node recording_config = config["record"];

        if (recording_config["first_frame"]) {
            m_first_frame = recording_config["first_frame"].as<int>();
        }
        if (recording_config["last_frame"]) {
            m_last_frame = recording_config["last_frame"].as<int>();
        }

        if (!recording_config["output"].IsSequence()) {
            spdlog::error("Error: No stream was selected for export!");
            exit(1);
        }

        YAML::Node output = recording_config["output"];

        for (YAML::const_iterator it=output.begin(); it!=output.end(); ++it) {
          if (it->as<std::string>() == "timestamp")
            m_export_timestamp = true;
          else if (it->as<std::string>() == "color")
            m_export_color = true;
          else if (it->as<std::string>() == "depth")
            m_export_depth = true;
          else if (it->as<std::string>() == "infrared")
            m_export_infrared = true;
          else if (it->as<std::string>() == "rgbd")
            m_export_rgbd = true;
          else if (it->as<std::string>() == "pointcloud")
            m_export_pointcloud = true;
          else
            spdlog::error("Invalid export type: [{0}].", it->as<std::string>());
        }

        m_input_directory = fs::path(recording_config["input_dir"].as<std::string>());
        m_output_directory = fs::path(recording_config["output_dir"].as<std::string>());

        if (!fs::exists(m_input_directory)) {
            spdlog::error("Input file: {0} not found!", m_input_directory.string());
            exit(1);
        }

        if (!fs::exists(m_output_directory)) {
            spdlog::error("Output directory: {0} does not exist", m_output_directory.string());
            exit(1);
        }

        YAML::Node feeds = recording_config["feeds"];

        for (YAML::const_iterator it=feeds.begin(); it!=feeds.end(); ++it) {
          fs::path input_feed = m_input_directory / it->as<std::string>();
          if (!fs::is_regular_file(input_feed)) {
            spdlog::error("Invalid filepath {0}", input_feed.string());
            exit(1);
          }
          m_input_feeds.push_back(input_feed);
        }
        spdlog::info("Successfully parsed configuration.");
    }

    int ExtractFramesMKV::exec() {
        Debug{} << "Core profile:" << GL::Context::current().isCoreProfile();
        Debug{} << "Context flags:" << GL::Context::current().flags();

        extract_mkv::Timesynchronizer ts{m_export_color, m_export_depth,
                                         m_export_infrared, m_export_rgbd,
                                         m_export_pointcloud, m_export_timestamp,
                                         m_first_frame, m_last_frame};
        ts.initialize_feeds(m_input_feeds, m_output_directory);
        ts.run();
        spdlog::info("Done.");
        return 0;
    }
}

MAGNUM_WINDOWLESSAPPLICATION_MAIN(Magnum::ExtractFramesMKV)
