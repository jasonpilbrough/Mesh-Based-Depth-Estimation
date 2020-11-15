#ifndef _SANDBOX_COLOURMAP
#define _SANDBOX_COLOURMAP

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


/* ADAPTED FROM OPENCV */

namespace sandbox{


  static cv::Mat linspace(float x0, float x1, int n);
  static cv::Mat argsort(cv::InputArray & _src, bool ascending=true);
  static void sortMatrixRowsByIndices(cv::InputArray & _src, cv::InputArray&  _indices, cv::OutputArray & _dst);
  static cv::Mat sortMatrixRowsByIndices(cv::InputArray & src, cv::InputArray & indices);

  template <typename _Tp> static
  cv::Mat interp1_(const cv::Mat& X_, const cv::Mat& Y_, const cv::Mat& XI);
  static cv::Mat interp1(cv::InputArray & _x, cv::InputArray & _Y, cv::InputArray & _xi);


class ColourMap {

    private:
      cv::Mat _lut;

    public:
        ColourMap();
        ColourMap(const int n);
        cv::Mat linear_colormap(cv::InputArray & X, cv::InputArray & r, cv::InputArray & g, cv::InputArray & b, cv::InputArray & xi);
        cv::Mat linear_colormap(cv::InputArray & X, cv::InputArray & r, cv::InputArray & g, cv::InputArray & b, const int n);
        cv::Mat linear_colormap(cv::InputArray & X, cv::InputArray & r, cv::InputArray & g, cv::InputArray & b, const float begin, const float end, const float n);

        void applyColourMap(cv::InputArray & src, cv::OutputArray & dst) const;
        void lookup(int lookup_value, cv::Vec3b & colour_out) const;
        void lookup2(const float v, cv::Vec3b & c) const;
        void lookup_alt(int lookup_value, cv::Vec3b & colour_out) const;


};

static ColourMap COLOUR_MAP; //instantiate a static variable as only need one instance







}

#endif
