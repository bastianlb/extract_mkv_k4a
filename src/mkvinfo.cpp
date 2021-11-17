#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

using namespace cv;

using namespace std;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


int main() {

    Mat image;

    namedWindow("Display window");

    VideoCapture cap(0);

    if (!cap.isOpened()) {

    cout << "cannot open camera";
    exit(1);

    }

    int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    //int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    // int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 30.0;
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter m_writer("test.avi", codec, fps, cv::Size(w, h), true);

    int frames = 0;
    while (frames < 100) {

        cap >> image;
        cv::Mat image(cv::Size(w, h), CV_8UC3, cv::Scalar(200, 150, 100));
        spdlog::info("Image type: {0}", type2str(image.type()));
        spdlog::info("Writing video frame...{0}, {1}", m_writer.get(cv::CAP_PROP_FRAME_WIDTH),
                m_writer.get(cv::CAP_PROP_FRAME_HEIGHT));
        m_writer.write(image);

        imshow("Display window", image);

        waitKey(25);
        frames++;

    }
    return 0;
}
