#ifndef _SANDBOX_PIPELINE
#define _SANDBOX_PIPELINE

#pragma once

#include <opencv2/core.hpp>
#include "framemanager.h"


namespace sandbox{

  struct Evaluation_stats{
    int img_count = 0;
    float total_MSE = 0;
    float total_MAE_Rel = 0;
    float count_delta10 = 0;
    float count_delta5 = 0;
    float count_delta1 = 0;
  };


  class Pipeline{
    private:
      FrameManager fm;
    public:
      Pipeline(const std::string & stereo_l_filepath, const std::string & stereo_r_filepath);
      Pipeline(const std::string & stereo_l_filepath, const std::string & stereo_r_filepath, const std::string & stereo_gnd_filepath);

      void loadCalib(cv::Mat & K1, cv::Mat & K2, cv::Mat & T1, cv::Mat & T2);
      void run();
      void evaluate(const cv::Mat & img_gnd, const cv::Mat & guess, Evaluation_stats & stats);
  };

}


#endif
