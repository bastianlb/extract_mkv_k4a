#include <opencv2/opencv.hpp>

#include <pcpd/datatypes.h>


int main(int argc, char* argv[])
{
    cv::Mat color_image = cv::imread("../test_data/0000000001_color.jpg");
    cv::Mat depth_image = cv::imread("../test_data/0000000001_depth.tiff");
    /*
    auto k4a_wrapper = std::make_shared<KPU::Kinect4AzureCaptureWrapper>("test");
    uint64_t depth_ts_ns = 12345667;
    size_t stride = (color_image.cols * color_image.elemSize());
    size_t nbytes = stride * color_image.rows;
    bool ret = k4a_wrapper->setColorImage(depth_ts_ns,
                                          depth_ts_ns, color_image.rows,
                                          color_image.cols, stride,
                                          (void*) (color_image.data),
                                          nbytes, pcpd::datatypes::PixelFormatType::BGRA);
    */




    return 0;
}
