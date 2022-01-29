#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "extract_mkv/extract_mkv_k4a.h"
#include "extract_mkv/filesystem.h"
#include "extract_mkv/kinect4azure_capture_wrapper.h"

using namespace extract_mkv;

void createAlphaImage(const cv::Mat& mat, cv::Mat_<cv::Vec4b>& dst)
{
  std::vector<cv::Mat> matChannels;
  cv::split(mat, matChannels);
  
  // create alpha channel
  cv::Mat alpha = matChannels.at(0) + matChannels.at(1) + matChannels.at(2);
  matChannels.push_back(alpha);

  cv::merge(matChannels, dst);
}

void getK4Acalib(k4a::calibration& calib) {
  fs::path in_file{"/media/narvis/atlas_4/archive_atlas_or01/03_animal_trials/210914_animal_trial_03/calibrations/calibration_01/recordings/cn01/capture-000000.mkv"};
  assert(fs::exists(in_file));
  K4AFrameExtractor extract{in_file.string(), "./", "test", ExportConfig{}};
  calib = extract.m_calibration;
}

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  cv::Mat color_image;
  cv::Mat _color_image = cv::imread("../test_data/color_frame_9.jpg");
  cv::cvtColor(_color_image, color_image, cv::COLOR_BGR2BGRA);

  cv::Mat depth_image = cv::imread("../test_data/depth_frame_9.tiff", CV_16UC1);
  auto k4a_wrapper = std::make_shared<KPU::Kinect4AzureCaptureWrapper>("test");
  uint64_t depth_ts_ns = 12345667;
  size_t col_w, col_h, depth_w, depth_h;
  col_w = color_image.cols;
  col_h = color_image.rows;
  std::cout << "Color image dim: " << col_w << ", " << col_h << std::endl;
  depth_w = depth_image.cols;
  depth_h = depth_image.rows;
  std::cout << "Depth image dim: " << depth_w << ", " << depth_h << std::endl;
  size_t stride = (col_w * color_image.elemSize());
  size_t nbytes = stride * col_h;
  bool ret = k4a_wrapper->setColorImage(depth_ts_ns,
                                        depth_ts_ns, col_w,
                                        col_h, stride,
                                        (void*) (color_image.data),
                                        nbytes, pcpd::datatypes::PixelFormatType::BGRA);
  stride = (depth_w * depth_image.elemSize());
  nbytes = stride * depth_h;
  bool rat = k4a_wrapper->setDepthImage(depth_ts_ns,
                                        depth_ts_ns,
                                        depth_w, depth_h,
                                        stride, (void*) (depth_image.data),
                                        nbytes);

  k4a::image input_depth_image = k4a_wrapper->capture_handle.get_depth_image();

  cv::Mat depth_image_buffer = cv::Mat(cv::Size(depth_w, depth_h), CV_16UC1,
                                       const_cast<void *>(static_cast<const void *>(input_depth_image.get_buffer())),
                                       static_cast<size_t>(input_depth_image.get_stride_bytes()));
  std::ostringstream ss;
  ss << "./test_depth.tiff";
  cv::imwrite(ss.str(), depth_image_buffer);

  k4a::image input_color_image = k4a_wrapper->capture_handle.get_color_image();
  int n_size = input_color_image.get_size();
  cv::Mat col_image_buffer = cv::Mat(cv::Size(col_w, col_h), CV_8UC4,
                                     const_cast<void*>(static_cast<const void *>(input_color_image.get_buffer())),
                                     static_cast<size_t>(input_color_image.get_stride_bytes()));
  ss.str(std::string());
  ss << "test_color.jpg";
  cv::imwrite(ss.str(), col_image_buffer);
  auto diff = depth_image != depth_image_buffer;
  auto s = sum(diff);
  EXPECT_TRUE(sum(diff) == cv::Scalar(0,0,0,0));
  EXPECT_TRUE(sum(depth_image != depth_image_buffer) == cv::Scalar(0));

  fs::path config_fname{"../test_data/camera_calibration.yml"};
  cv::FileStorage fs(config_fname, cv::FileStorage::READ);
  if (!fs.isOpened())
    {
      std::cerr << "failed to open " << config_fname.string() << std::endl;
      exit(1);
    }

  //std::cout << cfg_fs.depth2color_rotation << std::endl;
  // get depth intrinsics
  cv::Mat depth_mat, depth_distortion;
  fs["depth_camera_matrix"] >> depth_mat;
  fs["depth_distortion_coefficients"] >> depth_distortion;
  Eigen::VectorXf radial_dist(6);
  radial_dist[0] = depth_distortion.at<float>(0); 
  radial_dist[1] = depth_distortion.at<float>(1); 
  radial_dist[2] = depth_distortion.at<float>(4);
  radial_dist[3] = depth_distortion.at<float>(5); 
  radial_dist[4] = depth_distortion.at<float>(6); 
  radial_dist[5] = depth_distortion.at<float>(7);
  Eigen::Vector2f tangent_dist;
  tangent_dist[0] = depth_distortion.at<float>(2);
  tangent_dist[1] = depth_distortion.at<float>(3);

  pcpd::datatypes::IntrinsicParameters d_intrinsics{depth_mat.at<float>(0,0), depth_mat.at<float>(1,1),
                                                    depth_mat.at<float>(0,2), depth_mat.at<float>(1,2),
                                                    static_cast<unsigned int> (depth_w),
                                                    static_cast<unsigned int> (depth_h),
                                                    Eigen::Matrix3f{}, radial_dist, tangent_dist};

  cv::Mat color_mat, color_distortion;
  fs["color_camera_matrix"] >> color_mat;
  fs["color_distortion_coefficients"] >> color_distortion;
  Eigen::VectorXf col_radial_dist(6);
  col_radial_dist[0] = color_distortion.at<float>(0); 
  col_radial_dist[1] = color_distortion.at<float>(1); 
  col_radial_dist[2] = color_distortion.at<float>(4);
  col_radial_dist[3] = color_distortion.at<float>(5); 
  col_radial_dist[4] = color_distortion.at<float>(6); 
  col_radial_dist[5] = color_distortion.at<float>(7);
  Eigen::Vector2f col_tangent_dist;
  col_tangent_dist[0] = color_distortion.at<float>(2);
  col_tangent_dist[1] = color_distortion.at<float>(3);

  pcpd::datatypes::IntrinsicParameters c_intrinsics{color_mat.at<float>(0,0), color_mat.at<float>(1,1),
                                                    color_mat.at<float>(0,2), color_mat.at<float>(1,2),
                                                    static_cast<unsigned int> (col_w),
                                                    static_cast<unsigned int> (col_h),
                                                    Eigen::Matrix3f{}, col_radial_dist, col_tangent_dist};
  // get extrinsics color2depth
  cv::Mat r, t;
  fs["depth2color_rotation"] >> r;
  fs["depth2color_translation"] >> t;
  std::cout << t.at<double>(0,0) << std::endl;
  Eigen::Vector3f trans;
  trans[0] = t.at<float>(0);
  trans[1] = t.at<float>(1);
  trans[2] = t.at<float>(2);
  Eigen::Matrix3f rot;
  rot(0, 0) = r.at<float>(0, 0);
  rot(0, 1) = r.at<float>(0, 1);
  rot(0, 2) = r.at<float>(0, 2);
  rot(1, 0) = r.at<float>(1, 0);
  rot(1, 1) = r.at<float>(1, 1);
  rot(1, 2) = r.at<float>(1, 2);
  rot(2, 0) = r.at<float>(2, 0);
  rot(2, 1) = r.at<float>(2, 1);
  rot(2, 2) = r.at<float>(2, 2);


  //Eigen::Quaternion<float> rot;
  pcpd::datatypes::ExtrinsicParameters color2depth_transform{trans, Eigen::Quaternionf{rot}};
  pcpd::datatypes::ExtrinsicParameters camera_pose{};
  pcpd::datatypes::DeviceCalibration device_calibration{d_intrinsics, c_intrinsics, 
                                                        color2depth_transform, camera_pose, true};
  fs::path output_dir{"./"};
  k4a::calibration calibration;
  KPU::toK4A(device_calibration, calibration, 1);
  auto device_wrapper = std::make_shared<extract_mkv::K4ADeviceWrapper>();
  device_wrapper->calibration = calibration;
  auto rectify_maps = extract_mkv::process_calibration(calibration, output_dir);
  device_wrapper->rectify_maps = rectify_maps;

  k4a::calibration k4a_calib;
  getK4Acalib(k4a_calib);
  //device_wrapper->calibration = k4a_calib;

  extract_mkv::process_rgbd(k4a_wrapper->capture_handle.get_depth_image(),
                            col_w, col_h,
                            device_wrapper,
                            output_dir, 0);
  extract_mkv::process_pointcloud(k4a_wrapper->capture_handle.get_color_image(),
                                  k4a_wrapper->capture_handle.get_depth_image(),
                                  device_wrapper,
                                  output_dir,
                                  0);
}
