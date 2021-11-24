#include <fstream>              // File IO
#include <iostream>             // Terminal IO
#include <sstream>              // Stringstreams
#include <exception>
#include <algorithm>

#include <Corrade/configure.h>
#include <Corrade/Utility/Arguments.h>

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

#include "Magnum/GL/Context.h"

#include "yaml-cpp/yaml.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/daily_file_sink.h>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

#include "extract_mkv/timesync.h"
#include "extract_mkv/filesystem.h"

#ifdef WITH_PCPD
#include "extract_mkv/pcpd_file_exporter.h"
#endif

enum FILE_MODE {
    PCPD,
    K4A,
};

namespace Magnum {

    class ExtractFramesMKV : Platform::WindowlessApplication {
    public:
        explicit ExtractFramesMKV(const Arguments &arguments);

        int exec() override;

    private:
        Corrade::Utility::Arguments args;

        FILE_MODE m_file_mode {K4A};
        size_t m_first_frame {0};
        size_t m_last_frame {0};
        size_t m_skip_frames {1};

        extract_mkv::ExportConfig m_export_config{};
        bool m_timesync{false};
        bool m_enable_seek{false};

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
        else if (loglevel == "trace")
            spdlog::set_level(spdlog::level::trace);
        spdlog::set_pattern("[%H:%M:%S] %^[%l]%$ %l [thread %t] %v");
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_st>());
        sinks.push_back(std::make_shared<spdlog::sinks::daily_file_sink_st>("logfile", 23, 59));
        auto combined_logger = std::make_shared<spdlog::logger>("combined_logger", begin(sinks), end(sinks));
        //register it if you need to access it globally
        spdlog::register_logger(combined_logger);

        std::string config_path = args.value<std::string>("config");

        YAML::Node config = YAML::LoadFile(config_path);

        if (!config["record"]) {
            spdlog::error("Incorrectly formatted config, exiting."); 
            exit(1);
        }

        if (config["mode"]) {
            auto mode = config["mode"].as<std::string>();
            std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
            if (mode == std::string("pcpd")) {
                m_file_mode = FILE_MODE::PCPD;
            } else if (mode == std::string("k4a")) {
                m_file_mode = FILE_MODE::K4A;
            } else {
                spdlog::error("Bad file mode.. defaulting to K4A");
            }
        }

        YAML::Node recording_config = config["record"];
        // TODO: currently pcpd works only with start/end TS
        // k4a only with frame based..

        if (recording_config["start_ts"]) {
            m_export_config.start_ts = recording_config["start_ts"].as<int64_t>();
        }
        if (recording_config["end_ts"]) {
            m_export_config.end_ts = recording_config["end_ts"].as<int64_t>();
        }

        if (recording_config["first_frame"]) {
            m_first_frame = recording_config["first_frame"].as<int>();
        }
        if (recording_config["last_frame"]) {
            m_last_frame = recording_config["last_frame"].as<int>();
        }
        if (recording_config["skip_frames"]) {
            m_skip_frames = recording_config["skip_frames"].as<int>();
            assert(m_skip_frames > 0);
            m_export_config.skip_frames = m_skip_frames;
        }

        if (recording_config["max_frame_count"]) {
            int max_count = recording_config["max_frame_count"].as<int>();
            assert(max_count > 0);
            m_export_config.max_frames_exported = max_count;
        }

        if (!recording_config["output"].IsSequence()) {
            spdlog::error("Error: No stream was selected for export!");
            exit(1);
        }

        YAML::Node output = recording_config["output"];

        for (YAML::const_iterator it=output.begin(); it!=output.end(); ++it) {
          if (it->as<std::string>() == "timestamp")
            m_export_config.export_timestamp = true;
          else if (it->as<std::string>() == "color")
            m_export_config.export_color = true;
          else if (it->as<std::string>() == "color_video")
            m_export_config.export_color_video = true;
          else if (it->as<std::string>() == "depth")
            m_export_config.export_depth = true;
          else if (it->as<std::string>() == "infrared")
            m_export_config.export_infrared = true;
          else if (it->as<std::string>() == "rgbd")
            m_export_config.export_rgbd = true;
          else if (it->as<std::string>() == "pointcloud")
            m_export_config.export_pointcloud = true;
          else if (it->as<std::string>() == "bodypose")
            m_export_config.export_bodypose = true;
          else if (it->as<std::string>() == "align_clouds")
            m_export_config.align_clouds= true;
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
          if (m_file_mode == K4A) {
              if (!fs::is_regular_file(input_feed)) {
                spdlog::error("Invalid filepath {0}, should be a k4a mkv.", input_feed.string());
                exit(1);
              }
          } else if (m_file_mode == PCPD) {
              if (!fs::is_directory(input_feed)) {
                spdlog::error("Invalid filepath {0}, should be a directory.", input_feed.string());
                exit(1);
              }
          }
          m_input_feeds.push_back(input_feed);
        }

        m_timesync = config["timesync"].as<bool>();
        m_enable_seek = config["seek"].as<bool>();

        spdlog::info("Successfully parsed configuration.");
    }

    int ExtractFramesMKV::exec() {
        //Debug{} << "Core profile:" << GL::Context::current().isCoreProfile();
        //Debug{} << "Context flags:" << GL::Context::current().flags();

        if (m_file_mode == PCPD) {
#ifdef WITH_PCPD
            extract_mkv::TimesynchronizerPCPD ts{m_export_config};
            ts.initialize_feeds(m_input_feeds, m_output_directory);
            ts.run();
#else
            spdlog::error("Pcpd exporter not defined. Compile with WITH_PCPD=ON");
#endif
        } else if (m_file_mode == K4A) {
            extract_mkv::TimesynchronizerK4A ts{m_first_frame, m_last_frame,
                                                m_skip_frames, m_export_config, 
                                                m_timesync, m_enable_seek};
            ts.initialize_feeds(m_input_feeds, m_output_directory);
            ts.run();
        } else {
            spdlog::error("Invalid file mode");
        }
        spdlog::info("Done.");
        exit(0);
        return 0;
    }
}

MAGNUM_WINDOWLESSAPPLICATION_MAIN(Magnum::ExtractFramesMKV)
