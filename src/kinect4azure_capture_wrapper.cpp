#include <spdlog/spdlog.h>
#include "extract_mkv/kinect4azure_capture_wrapper.h"

namespace KPU {
    int pixelFormatToBits(pcpd::datatypes::PixelFormatType pixel_format)
    {
        switch(pixel_format)
        {
            case pcpd::datatypes::PixelFormatType::BGRA:
            case pcpd::datatypes::PixelFormatType::RGBA:
            case pcpd::datatypes::PixelFormatType::MJPEG:
                return 32;
            case pcpd::datatypes::PixelFormatType::RGB:
                return 24;
            case pcpd::datatypes::PixelFormatType::BGR:
                return 24;
            case pcpd::datatypes::PixelFormatType::DEPTH:
            case pcpd::datatypes::PixelFormatType::ZDEPTH:
                return 16;
            case pcpd::datatypes::PixelFormatType::LUMINANCE:
                return 8;
            default:
                return 0;
        }
    }

    Kinect4AzureCaptureWrapper::Kinect4AzureCaptureWrapper(std::string feed_name) : m_feed_name(feed_name) {
        capture_handle = k4a::capture::create();
    }

    bool Kinect4AzureCaptureWrapper::setColorImage(const uint64_t device_ts,
                       const uint64_t host_ts,
                       int out_width,
                       int out_height,
                       int out_stride,
                       void* data_ptr,
                       std::size_t data_size,
                       pcpd::datatypes::PixelFormatType pixel_format) {
        k4a_image_format_t format;
        switch (pixel_format) {
            case pcpd::datatypes::PixelFormatType::BGRA:
                format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
                break;
            case pcpd::datatypes::PixelFormatType::MJPEG:
                format = K4A_IMAGE_FORMAT_COLOR_MJPG;
                break;
            default:
                return false;
        }

        k4a::image image_handle;
        auto ret = convert_block_to_image(device_ts, host_ts, out_width, out_height, out_stride, data_ptr, data_size, image_handle, format);
        if (ret) {
            capture_handle.set_color_image(image_handle);
        } else {
            spdlog::error("Could not convert block to color image..");
        }
        return ret;
    }

    bool Kinect4AzureCaptureWrapper::setDepthImage(const uint64_t device_ts,
                       const uint64_t host_ts,
                       int out_width,
                       int out_height,
                       int out_stride,
                       void* data_ptr,
                       std::size_t data_size) {
        k4a::image image_handle;
        auto ret = convert_block_to_image(device_ts, host_ts, out_width, out_height, out_stride, data_ptr, data_size, image_handle, K4A_IMAGE_FORMAT_DEPTH16);
        if (ret) {
            capture_handle.set_depth_image(image_handle);
        } else {
            spdlog::error("Could not convert block to depth image..");
        }
        return ret;
    }

    bool Kinect4AzureCaptureWrapper::setInfraredImage(const uint64_t device_ts,
                          const uint64_t host_ts,
                          int out_width,
                          int out_height,
                          int out_stride,
                          void* data_ptr,
                          std::size_t data_size) {
        k4a::image image_handle;
        auto ret = convert_block_to_image(device_ts, host_ts, out_width, out_height, out_stride, data_ptr, data_size, image_handle, K4A_IMAGE_FORMAT_IR16);
        if (ret) {
            capture_handle.set_ir_image(image_handle);
        } else {
            spdlog::error("Could not convert block to IR image..");
        }
        return ret;
    }

    bool Kinect4AzureCaptureWrapper::convert_block_to_image(const uint64_t device_ts,
                                const uint64_t host_ts,
                                int out_width,
                                int out_height,
                                int out_stride,
                                void* data_ptr,
                                std::size_t data_size,
                                k4a::image& image_out,
                                k4a_image_format_t target_format) {
        if (data_size == 0) {
            spdlog::error("Kinect4AzureCaptureWrapper: error creating image from zero-length buffer");
            return false;
        }

        k4a_result_t result = K4A_RESULT_SUCCEEDED;
        assert(out_height >= 0 && out_width >= 0);

        switch (target_format)
        {
            case K4A_IMAGE_FORMAT_DEPTH16:
                break;
            case K4A_IMAGE_FORMAT_IR16:
                break;
            case K4A_IMAGE_FORMAT_COLOR_MJPG:
            case K4A_IMAGE_FORMAT_COLOR_BGRA32:
                // No format conversion is required, just use the buffer.
                break;
            default:
                result = K4A_RESULT_FAILED;
        }

        if (K4A_SUCCEEDED(result) && data_ptr != nullptr)
        {
            try {
                image_out = k4a::image::create_from_buffer(target_format,
                                                           out_width,
                                                           out_height,
                                                           out_stride,
                                                           static_cast<uint8_t*>(data_ptr),
                                                           data_size,
                                                           nullptr,
                                                           nullptr);
                uint64_t device_timestamp_usec = device_ts / 1000;
                image_out.set_timestamp(std::chrono::microseconds(device_timestamp_usec));
                // cannot be set via c++ interface .. not sure if this works..
                k4a_image_set_system_timestamp_nsec(image_out.handle(), host_ts);
            } catch (const k4a::error& e) {
                spdlog::error("Kinect4AzureCaptureWrapper: error creating image from buffer: {0}", e.what());
                result = K4A_RESULT_FAILED;
            }
        }

        return K4A_SUCCEEDED(result);
    }

    bool deserializeCalibrations(artekmed::schema::RecordingSchema::Reader& reader, 
            pcpd::datatypes::DeviceCalibration& device_calibration) {
        for(auto calibration : reader.getCalibrations())
            {
                std::string device_name {calibration.getName().cStr()};
                spdlog::info("Found calibration {0}", device_name);

                auto tm = pcpd::serialization::CapnprotoArchive::typeMap<pcpd::datatypes::DeviceCalibration>();
                auto calib_reader = calibration.getCalibration();

                tm.deserialize(calib_reader, device_calibration);
                return true;
            }
        return false;
    }


    bool intrinsicsToK4A(const pcpd::datatypes::IntrinsicParameters& params, k4a_calibration_camera_t& out) {
        out.intrinsics.parameters.param.fx = params.fov_x;
        out.intrinsics.parameters.param.fy = params.fov_y;
        out.intrinsics.parameters.param.cx = params.c_x;
        out.intrinsics.parameters.param.cy = params.c_y;
        out.intrinsics.parameters.param.codx = 0;
        out.intrinsics.parameters.param.cody = 0;
        out.intrinsics.parameters.param.k1 = params.radial_distortion(0);
        out.intrinsics.parameters.param.k2 = params.radial_distortion(1);
        out.intrinsics.parameters.param.k3 = params.radial_distortion(2);
        out.intrinsics.parameters.param.k4 = params.radial_distortion(3);
        out.intrinsics.parameters.param.k5 = params.radial_distortion(4);
        out.intrinsics.parameters.param.k6 = params.radial_distortion(5);
        out.intrinsics.parameters.param.p1 = params.tangential_distortion(0);
        out.intrinsics.parameters.param.p2 = params.tangential_distortion(1);
        // the extrinsics need to be initialzed properly
        float rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float trans[3] = {0, 0, 0};
        memcpy(&out.extrinsics.rotation, rot, 9 * sizeof(float));
        memcpy(&out.extrinsics.translation, trans, 3 * sizeof(float));

        out.resolution_width = static_cast<int>(params.width);
        out.resolution_height = static_cast<int>(params.height);
        // TODO: check this.. pcpd has 1.7f hardcoded in new version??
        out.metric_radius = 1.74f; // params.metric_radius;
        out.intrinsics.type = K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY;
        out.intrinsics.parameter_count = 14;
        return true;
    }
    bool extrinsicsToK4A(pcpd::datatypes::ExtrinsicParameters extrinsics, k4a_calibration_extrinsics_t& out, float units_per_meter) {
        auto trans = extrinsics.translation;
        auto rot = extrinsics.rotation.normalized().toRotationMatrix();
        out.translation[0] = trans(0) * units_per_meter;
        out.translation[1] = trans(1) * units_per_meter;
        out.translation[2] = trans(2) * units_per_meter;
        out.rotation[0] = rot(0, 0);
        out.rotation[1] = rot(0, 1);
        out.rotation[2] = rot(0, 2);
        out.rotation[3] = rot(1, 0);
        out.rotation[4] = rot(1, 1);
        out.rotation[5] = rot(1, 2);
        out.rotation[6] = rot(2, 0);
        out.rotation[7] = rot(2, 1);
        out.rotation[8] = rot(2, 2);
        return true;
    }
    bool toK4A(const pcpd::datatypes::DeviceCalibration& calibration, k4a::calibration& out, float units_per_meter) {
        bool result{true};
        result &= extrinsicsToK4A(calibration.color2depth_transform, out.extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH], units_per_meter);
        result &= intrinsicsToK4A(calibration.depth_parameters, out.depth_camera_calibration);
        result &= intrinsicsToK4A(calibration.color_parameters, out.color_camera_calibration);
        // calibration.camera_pose is not available in K4A structs
        out.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        out.color_resolution = K4A_COLOR_RESOLUTION_1536P;
        return result;
    }
}
