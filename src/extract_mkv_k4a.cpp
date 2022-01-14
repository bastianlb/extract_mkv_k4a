#include <iostream>

#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <k4arecord/playback.hpp>

#include <spdlog/spdlog.h>
#include <json/json.h>
#include <json/writer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "extract_mkv/extract_mkv_k4a.h"
#include "extract_mkv/filesystem.h"


namespace extract_mkv {

    K4AFrameExtractor::K4AFrameExtractor(std::string input_filename, std::string output_directory, 
            std::string feed_name, ExportConfig export_config) :
        m_input_filename(fs::path(input_filename)), m_output_directory(fs::path(output_directory)), 
        m_name(feed_name), m_export_config(export_config) {

        fs::create_directories(output_directory);
        m_dev = k4a::playback::open(m_input_filename.c_str());

        m_dev_config = m_dev.get_record_configuration();
        m_calibration = m_dev.get_calibration();

        k4a_record_configuration_t record_config = m_dev.get_record_configuration();
        m_timestamp_offset = std::chrono::microseconds(record_config.start_timestamp_offset_usec);

        // store calibration
        print_raw_calibration(m_calibration);
        m_rectify_maps = process_calibration(m_calibration, m_output_directory);

        if (export_config.export_pointcloud) {
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

        if (m_export_config.export_pointcloud || m_export_config.export_extrinsics) {
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
            Eigen::Quaternionf q{rot["w"].asFloat(), rot["x"].asFloat(),
                                 rot["y"].asFloat(), rot["z"].asFloat()};
            Eigen::Vector3f t{trans["m00"].asFloat(),
                              trans["m10"].asFloat(),
                              trans["m20"].asFloat()};
            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
            transform.block<3, 3>(0, 0) = q.toRotationMatrix();
            transform.block<3, 1>(0, 3) = t;

            // flip y and z from opengl to opencv convention
            Eigen::Matrix4f yz_transform = Eigen::Matrix4f::Identity();
            yz_transform(1,1) = -1.0;
            yz_transform(2,2) = -1.0;
            transform = yz_transform * transform * yz_transform.transpose();

            // flip back to a convention where Z is up.
            Eigen::Matrix4f swap_y_z = Eigen::Matrix4f::Zero();
            auto m = Eigen::AngleAxisf(-0.5*M_PI, Eigen::Vector3f::UnitX());
            swap_y_z.block<3, 3>(0, 0) = m.toRotationMatrix();
            swap_y_z(3, 3) = 1;

            transform = swap_y_z * transform;
            m_extrinsics.matrix() = transform;
            std::cout << "Cam " << m_name << " extrinsics: " << m_extrinsics.matrix() << std::endl;
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

    void K4AFrameExtractor::seek(int frame) {
        std::chrono::duration fps = std::chrono::seconds(get_fps());
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(frame * fps);
        if (timestamp.count() < m_dev.get_recording_length().count()) {
            spdlog::debug("Feed {0} seeking to timestamp: {1}ms", m_name, timestamp.count());
            // first get one capture..
            m_dev.get_next_capture(&m_capture);
            m_dev.seek_timestamp(timestamp, K4A_PLAYBACK_SEEK_DEVICE_TIME);
        } else {
            spdlog::error("Feed {0} unable to seek to timestamp: {1}ms", m_name, timestamp.count());
        }
    }

    void K4AFrameExtractor::next_capture() {
        std::scoped_lock lock(m_worker_lock);
        m_dev.get_next_capture(&m_capture);
        const k4a::image depth_image = m_capture.get_depth_image();
        if (depth_image.is_valid())
            m_last_depth_ts = depth_image.get_device_timestamp() + m_timestamp_offset;
        else
            m_last_depth_ts = std::chrono::microseconds(0);
    }

    void K4AFrameExtractor::record_timestamps(k4a::image depth_image, k4a::image color_image, int frame_counter) {
        m_tsss.str("");
        m_tsss.clear();
        if (!depth_image.is_valid() && !color_image.is_valid()) {
            spdlog::error("Failed to acquire images for cam {0} frame {1}", m_name, frame_counter);
        }
        if (m_export_config.export_timestamp) {
            int color_system_ts;
            int depth_system_ts;
            m_tsss << std::to_string(frame_counter) << ",";
            if (depth_image.is_valid())
                 depth_system_ts = depth_image.get_system_timestamp().count();
            else
                depth_system_ts = 0;
            if (color_image.is_valid())
                color_system_ts = color_image.get_system_timestamp().count();
            else
                color_system_ts = 0;
            m_tsss << m_last_depth_ts.count() << "," << depth_system_ts << ",";
            m_tsss << m_last_color_ts.count() << "," << color_system_ts << ",";
            // don't include IR timestamp export by default
            if (m_export_config.export_infrared) {
                const k4a::image ir_image = m_capture.get_ir_image();
                if (ir_image.is_valid()) {
                    int ir_device_ts = ir_image.get_device_timestamp().count();
                    int ir_system_ts = ir_image.get_system_timestamp().count();
                    m_tsss << ir_device_ts << "," << ir_system_ts << ",";
                }
            }
            m_timestamp_file << m_tsss.str() << std::endl;
        }

    }

    void K4AFrameExtractor::extract_frames(int frame_counter) {
        try {
            m_worker_lock.lock();
            // TODO: does this copy, or just get a reference..
            const k4a::image input_depth_image = m_capture.get_depth_image();
            const k4a::image input_color_image = m_capture.get_color_image();
            const k4a::image input_ir_image = m_capture.get_ir_image();
            m_worker_lock.unlock();
            std::shared_ptr<K4ADeviceWrapper> wrapper;
            wrapper->rectify_maps = m_rectify_maps;
            wrapper->calibration = m_calibration;
            spdlog::info("Processing {0} : {1}", m_name, frame_counter);
            record_timestamps(input_color_image, input_depth_image, frame_counter);
            if (m_export_config.export_depth && input_depth_image.is_valid()) {
                process_depth(input_depth_image, wrapper, m_output_directory, frame_counter);
            }

            if (m_export_config.export_color && input_color_image.is_valid()) {
                process_color(input_color_image, wrapper, m_output_directory, frame_counter);
            }

            if (m_export_config.export_infrared && input_ir_image.is_valid()) {
                process_ir(input_ir_image, wrapper, m_output_directory, frame_counter);
            }

            if (m_export_config.export_rgbd && input_depth_image.is_valid() && input_color_image.is_valid()) {
                process_rgbd(input_color_image, input_depth_image, wrapper, m_output_directory, frame_counter);
            }

            if (m_export_config.export_pointcloud && input_depth_image.is_valid() && input_color_image.is_valid()) {
                process_pointcloud(input_color_image, input_depth_image, wrapper, m_output_directory, frame_counter);
            }

            if (m_export_config.export_bodypose && input_depth_image.is_valid() && input_ir_image.is_valid()) {
                process_pose(wrapper, m_output_directory, frame_counter);
            }
        } catch (const extract_mkv::MissingDataException& e) {
          spdlog::error("Error during playback: {0}", e.what());
          m_worker_lock.unlock();
        } catch(const std::exception& e) {
          spdlog::error("Error during playback: {0}", e.what());
          m_worker_lock.unlock();
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

    int process_depth(k4a::image input_depth_image, std::shared_ptr<K4ADeviceWrapper> device_wrapper, fs::path output_directory, int frame_counter) {
        if (input_depth_image.is_valid()) {
            spdlog::debug("K4A processing depth image {0}", frame_counter);
            uint timestamp = input_depth_image.get_system_timestamp().count();
            int w = input_depth_image.get_width_pixels();
            int h = input_depth_image.get_height_pixels();
            spdlog::trace("Exporting depth image with width {0} and height {1}", w, h);

            if (input_depth_image.get_format() == K4A_IMAGE_FORMAT_DEPTH16) {
                cv::Mat image_buffer = cv::Mat(cv::Size(w, h), CV_16UC1,
                                               const_cast<void *>(static_cast<const void *>(input_depth_image.get_buffer())),
                                               static_cast<size_t>(input_depth_image.get_stride_bytes()));
                cv::Mat undistorted_image;
                cv::remap(image_buffer, undistorted_image, device_wrapper->rectify_maps.depth_map_x,
                          device_wrapper->rectify_maps.depth_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
                std::ostringstream ss;
                ss << std::setw(10) << std::setfill('0') << frame_counter << "_depth.tiff";
                std::string image_path = output_directory / ss.str();
                cv::imwrite(image_path, undistorted_image);
                std::ostringstream s;
                s << std::setw(10) << std::setfill('0') << frame_counter << "_distorted_depth.tiff";
                image_path = output_directory / s.str();
                cv::Mat image;
                image_buffer.convertTo(image, CV_8UC1);
                cv::imwrite(image_path, image);
            } else {
                spdlog::warn("Received depth frame with unexpected format: {0}", input_depth_image.get_format());
                throw MissingDataException();
            }
            return (int)timestamp;
        } else {
            return -1;
        }
    }

    int process_color(k4a::image input_color_image, std::shared_ptr<K4ADeviceWrapper> device_wrapper, fs::path output_directory, int frame_counter) {
        if (!input_color_image.is_valid()) {
            spdlog::warn("Color image invalid for {0}, frame {1}", output_directory, frame_counter);
            return -1;
        }
        spdlog::debug("K4A processing color image {0}", frame_counter);
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
            cv::remap(image_buffer, undistorted_image, device_wrapper->rectify_maps.color_map_x,
                      device_wrapper->rectify_maps.color_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

            spdlog::debug("Writing color image..");
            ss << std::setw(10) << std::setfill('0') << frame_counter << "_color.jpg";
            fs::path image_path = output_directory / ss.str();
            cv::imwrite(image_path, undistorted_image, compression_params);
            std::ostringstream s;
            s << std::setw(10) << std::setfill('0') << frame_counter << "_distorted_color.jpg";
            image_path = output_directory / s.str();
            // cv::imwrite(image_path, image_buffer, compression_params);
            cv::Mat diff;
            std::ostringstream sss;
            cv::absdiff(image_buffer, undistorted_image, diff);
            sss << std::setw(10) << std::setfill('0') << frame_counter << "_diff_color.jpg";
            image_path = output_directory / sss.str();
            // cv::imwrite(image_path, diff, compression_params);

        } else if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_MJPG) {
            int n_size = input_color_image.get_size();
            cv::Mat raw_data(1, n_size, CV_8UC1, (void*)(input_color_image.get_buffer()), input_color_image.get_size());
            image_buffer = cv::imdecode(raw_data, cv::IMREAD_COLOR);
            if ( image_buffer.data == NULL ) {
                // Error reading raw image data
                throw MissingDataException();
            }
            cv::remap(image_buffer, undistorted_image, device_wrapper->rectify_maps.color_map_x,
                      device_wrapper->rectify_maps.color_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
            ss << std::setw(10) << std::setfill('0') << frame_counter << "_color.jpg";
            fs::path image_path = output_directory / ss.str();
            cv::imwrite(image_path, undistorted_image, compression_params);
            std::ostringstream s;
            s << std::setw(10) << std::setfill('0') << frame_counter << "_distored_color.jpg";
            image_path = output_directory / s.str();
            // cv::imwrite(image_path, image_buffer, compression_params);
        } else {
            spdlog::warn("Received color frame with unexpected format: {0}",
                        input_color_image.get_format());
            throw MissingDataException();
        }
        return timestamp;
    }

    void process_rgbd(k4a::image input_color_image,
                      k4a::image input_depth_image,
                      std::shared_ptr<K4ADeviceWrapper> device_wrapper,
                      fs::path output_directory, int frame_counter) {

        int color_image_width_pixels = k4a_image_get_width_pixels(input_color_image.handle());
        int color_image_height_pixels = k4a_image_get_height_pixels(input_color_image.handle());

        if (!(input_color_image.is_valid() && input_depth_image.is_valid())) {
            spdlog::warn("Export RGBD requires depth and color image.");
            throw MissingDataException();
        }
        k4a_image_t transformed_depth_image;
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                     color_image_width_pixels,
                                                     color_image_height_pixels,
                                                     color_image_width_pixels * (u_int16_t)sizeof(uint16_t),
                                                     &transformed_depth_image))
        {
            spdlog::error("Failed to create transformed color image");
            k4a_image_release(transformed_depth_image);
            throw MissingDataException();
        }

        k4a_transformation_t transformation = k4a_transformation_create(&device_wrapper->calibration);
        if (K4A_RESULT_SUCCEEDED !=
                k4a_transformation_depth_image_to_color_camera(transformation, input_depth_image.handle(),
                                                               transformed_depth_image))
        {
            spdlog::error("Failed to compute transformed depth image");
            k4a_image_release(transformed_depth_image);
            k4a_transformation_destroy(transformation);
            throw MissingDataException();
        }
        std::ostringstream ss;
        ss << std::setw(10) << std::setfill('0') << frame_counter << "_rgbd.tiff";
        fs::path image_path = output_directory / ss.str();
        cv::Mat image_buffer = cv::Mat(cv::Size(color_image_width_pixels, color_image_height_pixels), CV_16UC1,
                                       const_cast<void *>(static_cast<const void *>(k4a_image_get_buffer(transformed_depth_image))),
                                       static_cast<size_t>(k4a_image_get_stride_bytes(transformed_depth_image)));
        double minVal;
        double maxVal;

        cv::minMaxLoc(image_buffer, &minVal, &maxVal);
        spdlog::info("RGBD min: {0}, max: {1}", minVal, maxVal);
        cv::Mat undistorted_image;
        // undistort using color image rectify maps?
        cv::remap(image_buffer, undistorted_image, device_wrapper->rectify_maps.color_map_x,
                  device_wrapper->rectify_maps.color_map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

        cv::imwrite(image_path, undistorted_image);
        std::ostringstream s;
        s << std::setw(10) << std::setfill('0') << frame_counter << "_distorted_rgbd.tiff";
        image_path = output_directory / s.str();
        // cv::imwrite(image_path, image_buffer);
        k4a_image_release(transformed_depth_image);
        k4a_transformation_destroy(transformation);
    };

    void process_pointcloud(k4a::image input_color_image,
                            k4a::image input_depth_image,
                            std::shared_ptr<K4ADeviceWrapper> device_wrapper,
                            fs::path output_directory,
                            int frame_counter,
                            bool align_clouds) {

        spdlog::trace("In process pointcloud");
        int color_image_width_pixels = input_color_image.get_width_pixels();
        int color_image_height_pixels = input_color_image.get_height_pixels();

        // transform color image into depth camera geometry
        int depth_image_width_pixels = input_depth_image.get_width_pixels();
        int depth_image_height_pixels = input_depth_image.get_height_pixels();
        k4a_image_t transformed_color_image = NULL;
        k4a::image color_image;
        cv::Mat result;
        spdlog::trace("Done initializing pointcloud");


        if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
            color_image = input_color_image;

        } else if (input_color_image.get_format() == K4A_IMAGE_FORMAT_COLOR_MJPG) {

            cv::Mat rawData(1, input_color_image.get_size(), CV_8SC1,
                            const_cast<void *>(static_cast<const void *>(input_color_image.get_buffer())));
            cv::Mat image_buffer = cv::imdecode(rawData, -cv::IMREAD_COLOR);

            cv::cvtColor(image_buffer, result, cv::COLOR_BGR2BGRA);
            color_image = k4a::image::create_from_buffer(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                         color_image_width_pixels,
                                                         color_image_height_pixels,
                                                         color_image_width_pixels * 4 * (int)sizeof(unsigned char),
                                                         result.data, result.total() * result.elemSize(), NULL, NULL);

        } else {
            spdlog::warn("Received color frame with unexpected format: {0}",
                         input_color_image.get_format());
            throw MissingDataException();
        }

        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                     depth_image_width_pixels,
                                                     depth_image_height_pixels,
                                                     depth_image_width_pixels * 4 * (int)sizeof(uint8_t),
                                                     &transformed_color_image))
        {
            spdlog::error("Failed to create transformed color image");
            k4a_image_release(transformed_color_image);
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
            k4a_image_release(transformed_color_image);
            k4a_image_release(point_cloud_image);
            throw MissingDataException();
        }

        k4a_transformation_t transformation = k4a_transformation_create(&device_wrapper->calibration);
        if (K4A_RESULT_SUCCEEDED !=
                k4a_transformation_color_image_to_depth_camera(transformation, input_depth_image.handle(), color_image.handle(), transformed_color_image))
        {
            spdlog::error("Failed to compute transformed color image");
            k4a_image_release(transformed_color_image);
            k4a_image_release(point_cloud_image);
            k4a_transformation_destroy(transformation);
            throw MissingDataException();
        }

        if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation,
                                                                                  input_depth_image.handle(),
                                                                                  K4A_CALIBRATION_TYPE_DEPTH,
                                                                                  point_cloud_image))
        {
            spdlog::error("Failed to compute point cloud");
            k4a_image_release(transformed_color_image);
            k4a_image_release(point_cloud_image);
            k4a_transformation_destroy(transformation);
            throw MissingDataException();
        }

        // TODO: use PCL? remove null points?
        spdlog::trace("Exporting pointcloud.");
        std::vector<color_point_t> points = image_to_pointcloud(point_cloud_image, transformed_color_image);
        /*
        TODO: not currently in framework
        if (align_clouds) {
            for (auto &point : points) {
                point.xyz = device_wrapper->world2camera * point.xyz;
            }
        }*/

        std::ostringstream ss;
        ss << std::setw(4) << std::setfill('0') << frame_counter << "_pointcloud.ply";
        fs::path ply_path = output_directory / ss.str();
        if (points.empty()) {
            spdlog::error("No points in pointcloud! {0}", ply_path.string());
        } else {
            tranformation_helpers_write_point_cloud(points, ply_path.c_str());
        }

        k4a_image_release(transformed_color_image);
        k4a_image_release(point_cloud_image);
        k4a_transformation_destroy(transformation);
    }

    void process_ir(k4a::image input_ir_image,
                   std::shared_ptr<K4ADeviceWrapper> device_wrapper,
                   fs::path output_directory, int frame_counter) {
        {
            if (input_ir_image.is_valid()) {

                int w = input_ir_image.get_width_pixels();
                int h = input_ir_image.get_height_pixels();
                spdlog::trace("Exporting IR image with width {0} and height {1}", w, h);

                if (input_ir_image.get_format() == K4A_IMAGE_FORMAT_IR16) {
                    cv::Mat image_buffer = cv::Mat(cv::Size(w, h), CV_16UC1,
                                                   const_cast<void *>(static_cast<const void *>(input_ir_image.get_buffer())),
                                                   static_cast<size_t>(input_ir_image.get_stride_bytes()));
                    //cv::Mat output;
                    //cv::cvtColor(image_buffer, output, cv::COLOR_GRAY2RGBA);
                    // cv::normalize(image_buffer, image_buffer, 0,255, cv::NORM_MINMAX, CV_8U);
                    uint timestamp = input_ir_image.get_system_timestamp().count();

                    std::ostringstream ss;
                    ss << std::setw(10) << std::setfill('0') << frame_counter << "_ir.tiff";
                    fs::path image_path = output_directory / ss.str();
                    cv::normalize(image_buffer, image_buffer, 0, 255, cv::NORM_MINMAX);
                    cv::imwrite(image_path, image_buffer);

                } else {
                    spdlog::warn("Received infrared frame with unexpected format: {0}",
                                 input_ir_image.get_format());
                    throw MissingDataException();
                }
                // return input_ir_image.get_device_timestamp().count();
            } else {
                throw MissingDataException();
            }
        }
    }

    void process_pose(std::shared_ptr<K4ADeviceWrapper> device_wrapper,
                      fs::path output_directory, int frame_counter) {
    };
    /*
    void process_pose(std::shared_ptr<K4ADeviceWrapper> device_wrapper,
                      fs::path output_directory, int frame_counter) {
        k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
        // tracker_config.sensor_orientation = K4ABT_SENSOR_ORIENTATION_FLIP180;
        auto tracker = k4abt::tracker::create(device_wrapper->calibration, tracker_config);

            if (tracker.enqueue_capture(device_wrapper->capture)) {
                k4abt::frame bodyFrame = tracker.pop_result();
                Json::Value bodies(Json::arrayValue);
                if (bodyFrame != nullptr && bodyFrame.get_num_bodies() > 0) {
                  for (size_t i = 0, idx = 0; i < bodyFrame.get_num_bodies(); ++i) {
                    k4abt_body_t k4abt_body = bodyFrame.get_body(i);
                    Json::Value joints(Json::arrayValue);
                    for (unsigned int j = 0; j < K4ABT_JOINT_COUNT; ++j, ++idx) {
                        auto joint = k4abt_body.skeleton.joints[j];
                        k4a_float3_t position = joint.position; // xyz
                        k4a_quaternion_t orientation = joint.orientation; // wxyz
                        k4abt_joint_confidence_level_t confidenceLevel = joint.confidence_level;

                        Json::Value bj;

                        bj["bodyId"] = static_cast<uint16_t>(k4abt_body.id);
                        bj["jointType"] = j;
                        bj["confidenceLevel"] = static_cast<uint8_t>(confidenceLevel);

                        // @todo: Unify K4A 2 PCPD Coordinate Transforms in k4a_utils !!

                        auto translation = Eigen::Vector3f(
                            position.v[0] / 1000.f,
                            position.v[1] / 1000.f,
                            position.v[2] / 1000.f
                        );

                        auto rotation = Eigen::Quaternion<float>(
                            orientation.v[0],
                            orientation.v[1],
                            orientation.v[2],
                            orientation.v[3]
                        );

                        Json::Value trans(Json::arrayValue);
                        trans.append(translation.x());
                        trans.append(translation.y());
                        trans.append(translation.z());
                        bj["translation"] = trans;

                        Json::Value rot(Json::arrayValue);
                        trans.append(rotation.w());
                        trans.append(rotation.x());
                        trans.append(rotation.y());
                        trans.append(rotation.z());
                        bj["rotation"] = rot;
                        joints.append(bj);
                    }
                  bodies.append(joints);
                }
            }
            std::ostringstream s;
            s << std::setw(10) << std::setfill('0') << frame_counter << "_bodyjoints.json";
            fs::path joint_path = output_directory / s.str();
            std::ofstream fout(joint_path);
            fout << bodies;
        }
    }
    */

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
