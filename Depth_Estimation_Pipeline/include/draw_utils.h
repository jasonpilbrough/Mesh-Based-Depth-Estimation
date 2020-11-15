#ifndef _SANDBOX_DRAW_UTILS
#define _SANDBOX_DRAW_UTILS

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "triangle/triangle.h"
#include "stereo_matching.h"


namespace sandbox{

/* ADAPTED FROM FLAME */

using Triangle = cv::Vec3i;

void drawShadedTriangleBarycentric(cv::Point p1, cv::Point p2, cv::Point p3, float v1, float v2, float v3, cv::Mat* img);
void interpolateMesh(const std::vector<Triangle> & triangles, const std::vector<sandbox::DepthFeaturePair> & support_points, cv::Mat & dense_map);




}


#endif
