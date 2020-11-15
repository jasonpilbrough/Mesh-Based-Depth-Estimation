#ifndef _SANDBOX_FRAME
#define _SANDBOX_FRAME

#pragma once

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iterator>
#include <cassert>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>


namespace sandbox{


class FrameManager{
  private:
    std::string dataset_path_left;
    std::string dataset_path_right;
    std::string dataset_path_gnd;
    cv::VideoCapture cap_left;
    cv::VideoCapture cap_right;
    cv::VideoCapture cap_gnd;


  public:
    FrameManager(const std::string & path_l, const std::string & path_r);
    FrameManager(const std::string & path_l, const std::string & path_r, const std::string & path_gnd);
    bool hasNext();
    void nextFrame(cv::Mat & img_l, cv::Mat & img_r);
    bool nextFrame(cv::Mat & img_l, cv::Mat & img_r, cv::Mat & img_gnd);
};

}


#endif
