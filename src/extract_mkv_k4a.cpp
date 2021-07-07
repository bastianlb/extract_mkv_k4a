#include <iostream>             // Terminal IO
#include <filesystem>

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

#include <spdlog/spdlog.h>
#include <json/json.h>
#include <json/writer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "extract_mkv_k4a.h"

namespace fs = std::filesystem;

namespace extract_mkv {

    K4AFrameExtractor::K4AFrameExtractor(std::string input_filename, std::string output_directory, 
            std::string feed_name, ExportConfig export_config) :
        m_input_filename(fs::path(input_filename)), m_output_directory(fs::path(output_directory)), 
        m_name(feed_name), m_export_config(export_config) {

        fs::create_directories(output_directory);
        m_dev = k4a::playback::open(m_input_filename.c_str());

        m_dev_config = m_dev.get_record_configuration();
        m_calibration = m_dev.get_calibration();

        // store calibration
        print_raw_calibration(m_calibration);
        m_rectify_maps = process_calibration(m_calibration, m_output_directory);

        if (export_config.export_pointcloud){
            spdlog::debug("Set color conversion to BGRA32 for pointcloud export");
            m_dev.set_color_conversion(K4A_IMAGE_FORMAT_COLOR_BGRA32);
        }

        if (m_export_config.export_timestamp) {
            m_timestamp_path = fs::path(m_output_directory) / "timestamp.csv";
            m_timestamp_file.open(m_timestamp_path.c_str(), std::ios::out);
            m_tsss << "frameindex,";
            m_tsss << "depth_dts,depth_sts,";
            m_tsss << "color_dts,color_sts,";
            if (m_export_config.export_depth) {
                m_tsss << "infrared_dts,infrared_sts";
            }
            m_timestamp_file << m_tsss.str() << std::endl;
        }
        if (m_export_config.align_clouds) {
            fs::path extrinsic_path = m_input_filename.parent_path() / "world2camera.json";
            std::ifstream ifs { extrinsic_path.c_str() };
            if (!ifs.is_open()) {
                spdlog::error("Could not find extrinsics for feed {0}", m_name);
                throw MissingDataException();
            }
            Json::Value doc;   // will contains the root value after parsing.
            Json::CharReaderBuilder builder;
            std::string errs;
            bool ok = Json::parseFromStream(builder, ifs, &doc, &errs);
            if ( !ok ) {
                spdlog::error("Could not parse extrinsics for feed {0}: {1}", m_name, errs);
                throw MissingDataException();
            }
            const Json::Value root = doc["value0"];
            const Json::Value rot = root["rotation"];
            const Json::Value trans = root["translation"];
            if (!root || !rot || !trans) {
                spdlog::error("Missing extrinsics for feed {0}", m_name);
                throw MissingDataException();
            }
            spdlog::info("Loaded extrinsic calibration file for feed {0}", m_name);
            Eigen::Quaternionf q{rot["w"].asFloat(), rot["x"].asFloat(), rot["y"].asFloat(), rot["z"].asFloat()};
            Eigen::Vector3f t{trans["m00"].asFloat(), trans["m10"].asFloat(), trans["m20"].asFloat()};
            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
            transform.block<3, 3>(0, 0) = q.toRotationMatrix();
            transform.block<3, 1>(0, 3) = t;

            // flip y and z from opengl to opencv convention
            Eigen::Matrix4f yz_transform = Eigen::Matrix4f::Identity();
            yz_transform(1,1) = -1.0;
            yz_transform(2,2) = -1.0;
            transform = (yz_transform * transform * yz_transform);
            // flip back to a convention where Z is up.
            Eigen::Matrix4f swap_y_z = Eigen::Matrix4f::Zero();
            swap_y_z(0, 0) = 1;
            swap_y_z(1, 2) = -1;
            swap_y_z(2, 1) = -1;
            swap_y_z(3, 3) = 1;
            transform = swap_y_z * transform;
            m_extrinsics.matrix() = transform;
            // write camera extrinsics
            std::ofstream file_id;
            fs::path filename = fs::path(output_directory) / "world2camera.json";
            file_id.open(filename.c_str(), std::ios::out);
            file_id << root << std::endl;
            file_id.close();
        }
    }

    K4AFrameExtractor::~K4AFrameExtractor() {
        m_dev.close();
    }

    void K4AFrameExtractor::next_capture(int frame_counter) {
        m_tsss.str("");
        m_tsss.clear();
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
        if (m_export_config.export_timestamp) {
            m_tsss << std::to_string(frame_counter) << ",";
            int depth_system_ts = depth_image.get_system_timestamp().count();
            m_tsss << m_last_depth_ts << "," << depth_system_ts << ",";
            int color_system_ts = depth_image.get_system_timestamp().count();
            m_tsss << m_last_color_ts << "," << color_system_ts << ",";
            // don't include IR timestamp export by default
            if (m_export_config.export_infrared) {
                const k4a::image ir_image = m_capture.get_ir_image();
                int ir_device_ts = ir_image.get_device_timestamp().count();
                int ir_system_ts = ir_image.get_system_timestamp().count();
                m_tsss << ir_device_ts << "," << ir_system_ts << ",";
            }
            m_timestamp_file << m_tsss.str() << std::endl;
        }
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

    int K4AFrameExtractor::process_depth(int frame_counter) {
        const k4a::image input_depth_image = m_capture.get_depth_image();
        uint timestamp = input_depth_image.get_system_timestamp().count();
        if (input_depth_image) {
            int w = input_depth_image.get_width_pixels();
            int h = input_depth_image.get_height_pixels();

            if (input_depth_image.get_format() == K4A_IMAGE_FORMAT_DEPTH16) {
                cv::Mat image_buffer = cv::Mat(cv::Size(w, h), CV_16UC1,
                                               const_cast<void *>(static_cast<const void *>(input_depth_image.get_buffer())),
                                               static_cast<size_t>(input_depth_image.get_stride_bytes()));
                cv::Mat undistorted_image;
                cv::remap(image_buffer, undistorted_image, m_rectify_maps.depth_map_x,
                          m_rectify_maps.depth_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
                std::ostringstream ss;
                ss << std::setw(10) << std::setfill('0') << frame_counter << "_depth.tiff";
                std::string image_path = m_output_directory / ss.str();
                cv::imwrite(image_path, undistorted_image);
                std::ostringstream s;
                s << std::setw(10) << std::setfill('0') << frame_counter << "_undistorted_depth.tiff";
                image_path = m_output_directory / s.str();
                cv::imwrite(image_path, image_buffer);
            } else {
                spdlog::warn("Received depth frame with unexpected format: {0}", input_depth_image.get_format());
                throw MissingDataException();
            }
            return (int)timestamp;
        } else {
            return -1;
        }
    }

    int K4AFrameExtractor::process_color(int frame_counter) {
        const k4a::image input_color_image = m_capture.get_color_image();
        {
            if (input_color_image) {
                cv::Mat undistorted_image;

                int w = input_color_image.get_width_pixels();
                int h = input_color_image.get_height_pixels();

                cv::Mat image_buffer;
                uint timestamp = input_color_image.get_system_timestamp().count();

                std::ostringstream ss;

                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(95);

                if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
                    cv::Mat image_buffer = cv::Mat(cv::Size(w, h), CV_8UC4,
                                                   const_cast<void*>(static_cast<const void *>(input_color_image.get_buffer())),
                                                   static_cast<size_t>(input_color_image.get_stride_bytes()));
                    cv::remap(image_buffer, undistorted_image, m_rectify_maps.color_map_x,
                              m_rectify_maps.color_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

                    ss << std::setw(10) << std::setfill('0') << frame_counter << "_color.jpg";
                    fs::path image_path = m_output_directory / ss.str();
                    cv::imwrite(image_path, undistorted_image, compression_params);

                } else if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_MJPG) {
                    int n_size = input_color_image.get_size();
                    cv::Mat raw_data(1, n_size, CV_8UC1, (void*)(input_color_image.get_buffer()), input_color_image.get_size());
                    image_buffer = cv::imdecode(raw_data, cv::IMREAD_COLOR);
                    if ( image_buffer.data == NULL ) {
                        // Error reading raw image data
                        throw MissingDataException();
                    }
                    cv::remap(image_buffer, undistorted_image, m_rectify_maps.color_map_x,
                              m_rectify_maps.color_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
                    ss << std::setw(10) << std::setfill('0') << frame_counter << "_color.jpg";
                    fs::path image_path = m_output_directory / ss.str();
                    cv::imwrite(image_path, undistorted_image, compression_params);
                    std::ostringstream s;
                    s << std::setw(10) << std::setfill('0') << frame_counter << "distored_color.jpg";
                    image_path = m_output_directory / s.str();
                    cv::imwrite(image_path, image_buffer, compression_params);
                } else {
                    spdlog::warn("Received color frame with unexpected format: {0}",
                                input_color_image.get_format());
                    throw MissingDataException();
                }
                return timestamp;
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
        cv::Mat undistorted_image;
        // undistort using color image rectify maps?
        cv::remap(image_buffer, undistorted_image, m_rectify_maps.color_map_x,
                  m_rectify_maps.color_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

        cv::imwrite(image_path, undistorted_image);
        std::ostringstream s;
        s << std::setw(10) << std::setfill('0') << frame_counter << "_distorted_rgbd.tiff";
        image_path = m_output_directory / s.str();
        cv::imwrite(image_path, image_buffer);
        k4a_image_release(transformed_depth_image);
        k4a_transformation_destroy(transformation);
    }

    void K4AFrameExtractor::process_pointcloud(int frame_counter) {

        spdlog::trace("In process pointcloud");
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
        spdlog::trace("Done initializing pointcloud");


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

        // TODO: use PCL? remove null points?
        std::vector<color_point_t> points = image_to_pointcloud(point_cloud_image, transformed_color_image);
        if (m_export_config.align_clouds) {
            for (auto &point : points) {
                point.xyz = m_extrinsics * point.xyz;
            }
        }

        std::ostringstream ss;
        ss << std::setw(4) << std::setfill('0') << frame_counter << "_pointcloud.ply";
        fs::path ply_path = m_output_directory / ss.str();
        tranformation_helpers_write_point_cloud(points, ply_path.c_str());

        k4a_image_release(transformed_color_image);
        k4a_image_release(point_cloud_image);
        k4a_transformation_destroy(transformation);
    }

    RectifyMaps process_calibration(k4a_calibration_t calibration, fs::path output_directory) {
        // converting the calibration data to OpenCV format
        // extrinsic transformation from depth to color camera
        cv::Mat se3 = cv::Mat(3, 3, CV_32FC1,
                              calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR].rotation);
        cv::Mat t_vec = cv::Mat(3, 1, CV_32F,
                                calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR].translation);

        // intrinsic parameters of the depth camera
        k4a_calibration_intrinsic_parameters_t *intrinsics = &calibration.depth_camera_calibration.intrinsics.parameters;
        // convert to opencv
        std::vector<float> _depth_camera_matrix = {
                intrinsics->param.fx, 0.f, intrinsics->param.cx, 0.f, intrinsics->param.fy, intrinsics->param.cy, 0.f,
                0.f, 1.f
        };
        cv::Mat depth_camera_matrix = cv::Mat(3, 3, CV_32F, &_depth_camera_matrix[0]);
        std::vector<float> _depth_dist_coeffs = {intrinsics->param.k1, intrinsics->param.k2, intrinsics->param.p1,
                                                 intrinsics->param.p2, intrinsics->param.k3, intrinsics->param.k4,
                                                 intrinsics->param.k5, intrinsics->param.k6};
        cv::Mat depth_dist_coeffs = cv::Mat(8, 1, CV_32F, &_depth_dist_coeffs[0]);

        // apply undistortion of Brown-conrady model
        int d_width = calibration.depth_camera_calibration.resolution_width;
        int d_height =  calibration.depth_camera_calibration.resolution_height;
        cv::Size depth_image_size = cv::Size(d_width, d_height);
        cv::Mat new_depth_camera_matrix = cv::getOptimalNewCameraMatrix(depth_camera_matrix, depth_dist_coeffs, depth_image_size, 0);  
        cv::Mat depth_map_x, depth_map_y;
        cv::initUndistortRectifyMap(depth_camera_matrix, depth_dist_coeffs, cv::Mat(), new_depth_camera_matrix,
                                    depth_image_size, CV_16SC2, depth_map_x, depth_map_y);

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
        int c_width = calibration.color_camera_calibration.resolution_width;
        int c_height = calibration.color_camera_calibration.resolution_height;
        cv::Size color_image_size = cv::Size(c_width, c_height);
        cv::Mat new_color_camera_matrix = cv::getOptimalNewCameraMatrix(color_camera_matrix, color_dist_coeffs,
                                                                        color_image_size, 0);  
        cv::Mat color_map_x, color_map_y;
        cv::initUndistortRectifyMap(color_camera_matrix, color_dist_coeffs, cv::Mat(), new_color_camera_matrix,
                                    color_image_size, CV_32F, color_map_x, color_map_y);

        // store configuration in output directory
        fs::path config_fname = output_directory / "camera_calibration.yml";
        cv::FileStorage cfg_fs(config_fname, cv::FileStorage::WRITE);
        cfg_fs << "depth_image_width" << calibration.depth_camera_calibration.resolution_width;
        cfg_fs << "depth_image_height" << calibration.depth_camera_calibration.resolution_height;
        cfg_fs << "depth_camera_matrix" << depth_camera_matrix;
        cfg_fs << "undistored_depth_camera_matrix" << new_depth_camera_matrix;
        cfg_fs << "depth_distortion_coefficients" << depth_dist_coeffs;

        cfg_fs << "color_image_width" << calibration.color_camera_calibration.resolution_width;
        cfg_fs << "color_image_height" << calibration.color_camera_calibration.resolution_height;
        cfg_fs << "color_camera_matrix" << color_camera_matrix;
        cfg_fs << "undistorted_color_camera_matrix" << new_color_camera_matrix;
        cfg_fs << "color_distortion_coefficients" << color_dist_coeffs;

        cfg_fs << "depth2color_translation" << t_vec;
        cfg_fs << "depth2color_rotation" << se3;
        RectifyMaps maps{depth_map_x, depth_map_y, color_map_x, color_map_y};
        return maps;
    }

    void print_raw_calibration(k4a_calibration_t& calibration)
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
