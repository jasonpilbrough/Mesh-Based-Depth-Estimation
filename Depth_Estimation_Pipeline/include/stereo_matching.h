#ifndef _SANDBOX_STEREO_MATCHING
#define _SANDBOX_STEREO_MATCHING


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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stereo/quasi_dense_stereo.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "params.h"



namespace sandbox{

  struct FeaturePair{
    cv::KeyPoint kp1;
    cv::KeyPoint kp2;
    FeaturePair(cv::KeyPoint keyp1,  cv::KeyPoint keyp2) : kp1(keyp1), kp2(keyp2) { }
  };

  struct DepthFeaturePair{
    cv::KeyPoint kp1;
    cv::KeyPoint kp2;
    float depth;

    DepthFeaturePair(cv::KeyPoint keyp1, cv::KeyPoint keyp2, float dep) : kp1(keyp1), kp2(keyp2), depth(dep) { }

    static bool dfp_sorter(DepthFeaturePair const& lhs, DepthFeaturePair const& rhs) {
      return lhs.depth < rhs.depth;
    }

  };



class StereoMatching {

  private:

  public:
    void block_matching(const Params & params,const cv::Mat & img1, const cv::Mat & img2, cv::Mat & disp_img);
    void semi_global_block_matching(const Params & params,const cv::Mat & img1, const cv::Mat & img2, cv::Mat & disp_img);
    void sparse_matching(const Params & params,const cv::Mat & img_l, const cv::Mat & img_r, const cv::Mat & disp_img, const float f, const float b, std::vector<DepthFeaturePair> & support_points, cv::Mat & support_points_img);

    void draw_depthmap(const Params & params, const std::vector<DepthFeaturePair> & support_points, cv::Mat & d_img);
};


}


#endif
