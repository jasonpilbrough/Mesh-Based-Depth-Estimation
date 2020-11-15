#ifndef _SANDBOX_REGULARISERS
#define _SANDBOX_REGULARISERS

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/core.hpp>

#include "triangle/triangle.h"
#include "delaunay.h"


namespace sandbox{


class Mesh_Regulariser{
  public:
      void run_TV(std::vector<DepthFeaturePair> &sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> &sparse_supports_out);
      void run_TGV(std::vector<DepthFeaturePair> &sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> &sparse_supports_out);
      void run_logTV(std::vector<DepthFeaturePair> &sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> &sparse_supports_out);
      void run_logTGV(std::vector<DepthFeaturePair> &sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> &sparse_supports_out);
};





}

#endif
