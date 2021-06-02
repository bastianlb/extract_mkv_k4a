#include <iostream>             // Terminal IO
#include <filesystem>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

#include <spdlog/spdlog.h>

#include <Corrade/Utility/Directory.h>
#include <Corrade/Containers/ArrayView.h>

#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/imgcodecs.hpp>   // Include OpenCV API

#include "extract_mkv_k4a.h"
#include "transformation_helpers.h"

namespace fs = std::filesystem;

namespace extract_mkv {

    K4AFrameExtractor::K4AFrameExtractor(std::string input_filename, std::string output_directory, std::string feed_name) :
        m_input_filename(fs::path(input_filename)), m_output_directory(fs::path(output_directory)), m_name(feed_name) {

        fs::create_directories(output_directory);

        m_dev = k4a::playback::open(m_input_filename.c_str());

        m_dev_config = m_dev.get_record_configuration();

        m_calibration = m_dev.get_calibration();

        // store calibration
        print_calibration(m_calibration);
        save_calibration(m_calibration, m_output_directory);

        /*if (m_export_pointcloud) {
            spdlog::debug("Set color conversion to BGRA32 for pointcloud export");
            m_dev.set_color_conversion(K4A_IMAGE_FORMAT_COLOR_BGRA32);
        }*/
        m_timestamp_path = fs::path(m_output_directory) / "timestamp.csv";
        if (m_export_timestamp) {
            m_tsss << "frameindex,depth_dts,depth_sts,color_dts,color_sts,infrared_dts,infrared_sts\n";
        }
        /* record frameindex
        tsss.str("");
        tsss.clear();
        tsss << std::to_string(frame_counter) << ",";
        */
    }

    uint8_t K4AFrameExtractor::get_fps() {
        // https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/structk4a__device__configuration__t.html
        k4a_fps_t fps = m_dev_config.camera_fps;
        if (fps == 0) {
            return 5;
        } else if (fps == 1) {
            return 15;
        } else if (fps == 2) {
            return 30;
        } else {
            spdlog::error("Invalid camera fps {0}", fps);
            throw 1;
            return -1;
        }
    }

    void K4AFrameExtractor::next_capture() {
        m_dev.get_next_capture(&m_capture);
        const k4a::image depth_image = m_capture.get_depth_image();
        if (depth_image)
            m_last_depth_ts = depth_image.get_device_timestamp().count();
        else
            m_last_depth_ts = -10e6;
        const k4a::image color_image = m_capture.get_depth_image();
        if (color_image)
            m_last_color_ts = color_image.get_device_timestamp().count();
        else
            m_last_color_ts = -10e6;
    }


    k4a::capture K4AFrameExtractor::get_capture_handle() {
        return m_capture;
    }

    K4AFrameExtractor::~K4AFrameExtractor() {
        m_dev.close();
    }

    int K4AFrameExtractor::process_depth(int frame_counter) {
        const k4a::image input_depth_image = m_capture.get_depth_image();
        if (input_depth_image) {

            int w = input_depth_image.get_width_pixels();
            int h = input_depth_image.get_height_pixels();

            if (input_depth_image.get_format() == K4A_IMAGE_FORMAT_DEPTH16) {
                cv::Mat image_buffer = cv::Mat(cv::Size(w, h), CV_16UC1,
                                               const_cast<void *>(static_cast<const void *>(input_depth_image.get_buffer())),
                                               static_cast<size_t>(input_depth_image.get_stride_bytes()));
                uint timestamp = input_depth_image.get_system_timestamp().count();

                std::ostringstream ss;
                ss << std::setw(10) << std::setfill('0') << frame_counter << "_depth.tiff";
                std::string image_path = m_output_directory / ss.str();
                cv::imwrite(image_path, image_buffer);

            } else {
                spdlog::warn("Received depth frame with unexpected format: {0}", input_depth_image.get_format());
                throw MissingDataException();
            }
            return input_depth_image.get_device_timestamp().count();
        } else {
            return -1;
        }
    }

    int K4AFrameExtractor::process_color(int frame_counter) {
        const k4a::image input_color_image = m_capture.get_color_image();
        {
            if (input_color_image) {

                int w = input_color_image.get_width_pixels();
                int h = input_color_image.get_height_pixels();

                cv::Mat image_buffer;
                uint timestamp;

                std::ostringstream ss;
                if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
                    cv::Mat image_buffer = cv::Mat(cv::Size(w, h), CV_8UC4,
                                                   const_cast<void*>(static_cast<const void *>(input_color_image.get_buffer())),
                                                   static_cast<size_t>(input_color_image.get_stride_bytes()));
                    timestamp = input_color_image.get_system_timestamp().count();

                    std::vector<int> compression_params;
                    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                    compression_params.push_back(95);

                    ss << std::setw(10) << std::setfill('0') << frame_counter << "_color.jpg";
                    fs::path image_path = m_output_directory / ss.str();
                    cv::imwrite(image_path, image_buffer, compression_params);

                } else if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_MJPG) {
                    // TODO: cast as ctype array and avoid corrade?
                    auto rawData = Corrade::Containers::ArrayView<uint8_t>(const_cast<uint8_t *>(input_color_image.get_buffer()), input_color_image.get_size());
                    ss << std::setw(10) << std::setfill('0') << frame_counter << "_color.jpg";
                    std::string image_path = Corrade::Utility::Directory::join(m_output_directory, ss.str());
                    Corrade::Utility::Directory::write(image_path, rawData);
                } else {
                    spdlog::warn("Received color frame with unexpected format: {0}",
                                input_color_image.get_format());
                    throw MissingDataException();
                }
                return input_color_image.get_device_timestamp().count();
            } else {
                return -1;
            }
        }
    }

    int K4AFrameExtractor::process_ir(int frame_counter) {
        const k4a::image input_ir_image = m_capture.get_ir_image();
        {
            if (input_ir_image) {

                int w = input_ir_image.get_width_pixels();
                int h = input_ir_image.get_height_pixels();

                if (input_ir_image.get_format() == K4A_IMAGE_FORMAT_IR16) {
                    cv::Mat image_buffer = cv::Mat(cv::Size(w, h), CV_16UC1,
                                                   const_cast<void *>(static_cast<const void *>(input_ir_image.get_buffer())),
                                                   static_cast<size_t>(input_ir_image.get_stride_bytes()));
                    uint timestamp = input_ir_image.get_system_timestamp().count();

                    std::ostringstream ss;
                    ss << std::setw(10) << std::setfill('0') << frame_counter << "_ir.tiff";
                    fs::path image_path = m_output_directory / ss.str();
                    cv::imwrite(image_path, image_buffer);

                } else {
                    spdlog::warn("Received infrared frame with unexpected format: {0}",
                                 input_ir_image.get_format());
                    throw MissingDataException();
                }
                return input_ir_image.get_device_timestamp().count();
            } else {
                return -1;
            }
        }
    }

    void K4AFrameExtractor::process_rgbd(int frame_counter) {

        const k4a::image input_depth_image = m_capture.get_depth_image();
        const k4a::image input_color_image = m_capture.get_color_image();

        int color_image_width_pixels = k4a_image_get_width_pixels(input_color_image.handle());
        int color_image_height_pixels = k4a_image_get_height_pixels(input_color_image.handle());

        if (!(input_color_image && input_depth_image)) {
            spdlog::warn("Export RGBD requires depth and color image.");
            throw MissingDataException();
        }
        k4a_image_t transformed_depth_image;
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                     color_image_width_pixels,
                                                     color_image_height_pixels,
                                                     color_image_width_pixels * (int)sizeof(uint16_t),
                                                     &transformed_depth_image))
        {
            spdlog::error("Failed to create transformed color image");
            throw MissingDataException();
        }

        k4a_transformation_t transformation = k4a_transformation_create(&m_calibration);
        if (K4A_RESULT_SUCCEEDED !=
                k4a_transformation_depth_image_to_color_camera(transformation, input_depth_image.handle(),
                                                               transformed_depth_image))
        {
            spdlog::error("Failed to compute transformed depth image");
            throw MissingDataException();
        }
        std::ostringstream ss;
        ss << std::setw(10) << std::setfill('0') << frame_counter << "_rgbd.tiff";
        fs::path image_path = m_output_directory / ss.str();
        cv::Mat image_buffer = cv::Mat(cv::Size(color_image_width_pixels, color_image_height_pixels), CV_16UC1,
                                       const_cast<void *>(static_cast<const void *>(k4a_image_get_buffer(transformed_depth_image))),
                                       static_cast<size_t>(k4a_image_get_stride_bytes(transformed_depth_image)));

        cv::imwrite(image_path, image_buffer);
        k4a_image_release(transformed_depth_image);
        k4a_transformation_destroy(transformation);
    }

    void K4AFrameExtractor::process_pointcloud(int frame_counter) {

        const k4a::image input_depth_image = m_capture.get_depth_image();
        const k4a::image input_color_image = m_capture.get_color_image();
        int color_image_width_pixels = k4a_image_get_width_pixels(input_color_image.handle());
        int color_image_height_pixels = k4a_image_get_height_pixels(input_color_image.handle());

        k4a_transformation_t transformation = k4a_transformation_create(&m_calibration);
        // transform color image into depth camera geometry
        int depth_image_width_pixels = k4a_image_get_width_pixels(m_capture.get_depth_image().handle());
        int depth_image_height_pixels = k4a_image_get_height_pixels(m_capture.get_depth_image().handle());
        k4a_image_t transformed_color_image = NULL;
        k4a::image color_image;
        cv::Mat result;


        if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
            color_image = m_capture.get_color_image();

        } else if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_MJPG) {

            cv::Mat rawData(1, m_capture.get_color_image().get_size(), CV_8SC1,
                            const_cast<void *>(static_cast<const void *>(m_capture.get_color_image().get_buffer())));
            cv::Mat image_buffer = cv::imdecode(rawData, -cv::IMREAD_COLOR);

            cv::cvtColor(image_buffer, result, cv::COLOR_BGR2BGRA);
            color_image = k4a::image::create_from_buffer(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                         color_image_width_pixels,
                                                         color_image_height_pixels,
                                                         color_image_width_pixels * 4 * (int)sizeof(unsigned char),
                                                         result.data, result.total() * result.elemSize(), NULL, NULL);

        } else {
            spdlog::warn("Received color frame with unexpected format: {0}",
                        m_capture.get_color_image().get_format());
            throw MissingDataException();
        }



        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                     depth_image_width_pixels,
                                                     depth_image_height_pixels,
                                                     depth_image_width_pixels * 4 * (int)sizeof(uint8_t),
                                                     &transformed_color_image))
        {
            spdlog::error("Failed to create transformed color image");
            throw MissingDataException();
        }

        k4a_image_t point_cloud_image = NULL;
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                                     depth_image_width_pixels,
                                                     depth_image_height_pixels,
                                                     depth_image_width_pixels * 3 * (int)sizeof(int16_t),
                                                     &point_cloud_image))
        {
            spdlog::error("Failed to create point cloud image");
            throw MissingDataException();
        }

        if (K4A_RESULT_SUCCEEDED !=
                k4a_transformation_color_image_to_depth_camera(transformation, m_capture.get_depth_image().handle(), color_image.handle(), transformed_color_image))
        {
            spdlog::error("Failed to compute transformed color image");
            throw MissingDataException();
        }

        if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation,
                                                                                  m_capture.get_depth_image().handle(),
                                                                                  K4A_CALIBRATION_TYPE_DEPTH,
                                                                                  point_cloud_image))
        {
            spdlog::error("Failed to compute point cloud");
            throw MissingDataException();
        }

        std::ostringstream ss;
        ss << std::setw(10) << std::setfill('0') << frame_counter << "_pointcloud.ply";
        fs::path ply_path = m_output_directory / ss.str();

        tranformation_helpers_write_point_cloud(point_cloud_image, transformed_color_image, ply_path.c_str());

        k4a_image_release(transformed_color_image);
        k4a_image_release(point_cloud_image);
        k4a_transformation_destroy(transformation);
    }

    void save_calibration(k4a_calibration_t calibration, fs::path output_directory) {
        // from Kinect SDK ...

        // converting the calibration data to OpenCV format
        // extrinsic transformation from color to depth camera
        cv::Mat se3 = cv::Mat(3, 3, CV_32FC1,
                              calibration.extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH].rotation);
        cv::Mat t_vec = cv::Mat(3, 1, CV_32F,
                                calibration.extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH].translation);

        // intrinsic parameters of the depth camera
        k4a_calibration_intrinsic_parameters_t *intrinsics = &calibration.depth_camera_calibration.intrinsics.parameters;
        std::vector<float> _depth_camera_matrix = {
                intrinsics->param.fx, 0.f, intrinsics->param.cx, 0.f, intrinsics->param.fy, intrinsics->param.cy, 0.f,
                0.f, 1.f
        };
        cv::Mat depth_camera_matrix = cv::Mat(3, 3, CV_32F, &_depth_camera_matrix[0]);
        std::vector<float> _depth_dist_coeffs = {intrinsics->param.k1, intrinsics->param.k2, intrinsics->param.p1,
                                                 intrinsics->param.p2, intrinsics->param.k3, intrinsics->param.k4,
                                                 intrinsics->param.k5, intrinsics->param.k6};
        cv::Mat depth_dist_coeffs = cv::Mat(8, 1, CV_32F, &_depth_dist_coeffs[0]);

        // intrinsic parameters of the color camera
        intrinsics = &calibration.color_camera_calibration.intrinsics.parameters;
        std::vector<float> _color_camera_matrix = {
                intrinsics->param.fx, 0.f, intrinsics->param.cx, 0.f, intrinsics->param.fy, intrinsics->param.cy, 0.f,
                0.f, 1.f
        };
        cv::Mat color_camera_matrix = cv::Mat(3, 3, CV_32F, &_color_camera_matrix[0]);
        std::vector<float> _color_dist_coeffs = {intrinsics->param.k1, intrinsics->param.k2, intrinsics->param.p1,
                                                 intrinsics->param.p2, intrinsics->param.k3, intrinsics->param.k4,
                                                 intrinsics->param.k5, intrinsics->param.k6};
        cv::Mat color_dist_coeffs = cv::Mat(8, 1, CV_32F, &_color_dist_coeffs[0]);

        // store configuration in output directory
        fs::path config_fname = output_directory / "camera_calibration.yml";
        cv::FileStorage cfg_fs(config_fname, cv::FileStorage::WRITE);
        cfg_fs << "depth_image_width" << calibration.depth_camera_calibration.resolution_width;
        cfg_fs << "depth_image_height" << calibration.depth_camera_calibration.resolution_height;
        cfg_fs << "depth_camera_matrix" << depth_camera_matrix;
        cfg_fs << "depth_distortion_coefficients" << depth_dist_coeffs;

        cfg_fs << "color_image_width" << calibration.color_camera_calibration.resolution_width;
        cfg_fs << "color_image_height" << calibration.color_camera_calibration.resolution_height;
        cfg_fs << "color_camera_matrix" << color_camera_matrix;
        cfg_fs << "color_distortion_coefficients" << color_dist_coeffs;

        cfg_fs << "depth2color_translation" << t_vec;
        cfg_fs << "depth2color_rotation" << se3;
    }

    void print_calibration(k4a_calibration_t& calibration)
    {
        using namespace std;

        {
            cout << "Depth camera:" << endl;
            auto calib = calibration.depth_camera_calibration;

            cout << "resolution width: " << calib.resolution_width << endl;
            cout << "resolution height: " << calib.resolution_height << endl;
            cout << "principal point x: " << calib.intrinsics.parameters.param.cx << endl;
            cout << "principal point y: " << calib.intrinsics.parameters.param.cy << endl;
            cout << "focal length x: " << calib.intrinsics.parameters.param.fx << endl;
            cout << "focal length y: " << calib.intrinsics.parameters.param.fy << endl;
            cout << "radial distortion coefficients:" << endl;
            cout << "k1: " << calib.intrinsics.parameters.param.k1 << endl;
            cout << "k2: " << calib.intrinsics.parameters.param.k2 << endl;
            cout << "k3: " << calib.intrinsics.parameters.param.k3 << endl;
            cout << "k4: " << calib.intrinsics.parameters.param.k4 << endl;
            cout << "k5: " << calib.intrinsics.parameters.param.k5 << endl;
            cout << "k6: " << calib.intrinsics.parameters.param.k6 << endl;
            cout << "center of distortion in Z=1 plane, x: " << calib.intrinsics.parameters.param.codx << endl;
            cout << "center of distortion in Z=1 plane, y: " << calib.intrinsics.parameters.param.cody << endl;
            cout << "tangential distortion coefficient x: " << calib.intrinsics.parameters.param.p1 << endl;
            cout << "tangential distortion coefficient y: " << calib.intrinsics.parameters.param.p2 << endl;
            cout << "metric radius: " << calib.intrinsics.parameters.param.metric_radius << endl;
        }

        {
            cout << "Color camera:" << endl;
            auto calib = calibration.color_camera_calibration;

            cout << "resolution width: " << calib.resolution_width << endl;
            cout << "resolution height: " << calib.resolution_height << endl;
            cout << "principal point x: " << calib.intrinsics.parameters.param.cx << endl;
            cout << "principal point y: " << calib.intrinsics.parameters.param.cy << endl;
            cout << "focal length x: " << calib.intrinsics.parameters.param.fx << endl;
            cout << "focal length y: " << calib.intrinsics.parameters.param.fy << endl;
            cout << "radial distortion coefficients:" << endl;
            cout << "k1: " << calib.intrinsics.parameters.param.k1 << endl;
            cout << "k2: " << calib.intrinsics.parameters.param.k2 << endl;
            cout << "k3: " << calib.intrinsics.parameters.param.k3 << endl;
            cout << "k4: " << calib.intrinsics.parameters.param.k4 << endl;
            cout << "k5: " << calib.intrinsics.parameters.param.k5 << endl;
            cout << "k6: " << calib.intrinsics.parameters.param.k6 << endl;
            cout << "center of distortion in Z=1 plane, x: " << calib.intrinsics.parameters.param.codx << endl;
            cout << "center of distortion in Z=1 plane, y: " << calib.intrinsics.parameters.param.cody << endl;
            cout << "tangential distortion coefficient x: " << calib.intrinsics.parameters.param.p1 << endl;
            cout << "tangential distortion coefficient y: " << calib.intrinsics.parameters.param.p2 << endl;
            cout << "metric radius: " << calib.intrinsics.parameters.param.metric_radius << endl;
        }

        auto extrinsics = calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
        cout << "depth2color translation: (" << extrinsics.translation[0] << "," << extrinsics.translation[1] << "," << extrinsics.translation[2] << ")" << endl;
        cout << "depth2color rotation: |" << extrinsics.rotation[0] << "," << extrinsics.rotation[1] << "," << extrinsics.rotation[2] << "|" << endl;
        cout << "                      |" << extrinsics.rotation[3] << "," << extrinsics.rotation[4] << "," << extrinsics.rotation[5] << "|" << endl;
        cout << "                      |" << extrinsics.rotation[6] << "," << extrinsics.rotation[7] << "," << extrinsics.rotation[8] << "|" << endl;

    }
} // namespace extract mkv
