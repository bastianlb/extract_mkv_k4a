#pragma once

#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <chrono>
#include "pcpd/record/enumerations.h"
#include "pcpd/datatypes/calibration.h"
#include "pcpd/datatypes/enumerations.h"
#include "pcpd/record/mkv/matroska_read.h"
#include "serialization/capnproto_serialization.h"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "pcpd/config.h"
#include "pcpd/rttr_registration.h"
#include "pcpd/record/enumerations.h"


namespace KPU {
    int pixelFormatToBits(pcpd::datatypes::PixelFormatType pixel_format);
    bool deserializeCalibrations(artekmed::schema::RecordingSchema::Reader&,
                                 pcpd::datatypes::DeviceCalibration&);
    bool intrinsicsToK4A(const pcpd::datatypes::IntrinsicParameters& params, k4a_calibration_camera_t& out);
    bool extrinsicsToK4A(pcpd::datatypes::ExtrinsicParameters extrinsics, k4a_calibration_extrinsics_t& out, float units_per_meter=1000);
    bool toK4A(const pcpd::datatypes::DeviceCalibration& calibration, k4a::calibration& out, float units_per_meter=1000);

    struct Kinect4AzureCaptureWrapper {

        Kinect4AzureCaptureWrapper(std::string);
        ~Kinect4AzureCaptureWrapper() = default;

        bool setColorImage(uint64_t device_ts,
                            uint64_t host_ts,
                            int out_width,
                            int out_height,
                            int out_stride,
                            void* data_ptr,
                            std::size_t data_size,
                            pcpd::datatypes::PixelFormatType pixel_format);

        bool setDepthImage(uint64_t device_ts,
                            uint64_t host_ts,
                            int out_width,
                            int out_height,
                            int out_stride,
                            void* data_ptr,
                            std::size_t data_size);

        bool setInfraredImage(uint64_t device_ts,
                               uint64_t host_ts,
                               int out_width,
                               int out_height,
                               int out_stride,
                               void* data_ptr,
                               std::size_t data_size);

        bool convert_block_to_image(uint64_t device_ts,
                                    uint64_t host_ts,
                                    int out_width,
                                    int out_height,
                                    int out_stride,
                                    void* data_ptr,
                                    std::size_t data_size,
                                    k4a::image& image_out,
                                    k4a_image_format_t target_format);


        k4a::capture capture_handle{};
        int frame_id;
        std::string m_feed_name;
    };
}
