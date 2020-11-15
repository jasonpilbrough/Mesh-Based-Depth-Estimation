#ifndef _SANDBOX_DELAUNAY
#define _SANDBOX_DELAUNAY

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "triangle/triangle.h"
#include "stereo_matching.h"


namespace sandbox{

/* ADAPTED FROM FLAME */

using Vertex = cv::Point2f;
using Triangle = cv::Vec3i;
using Edge = cv::Vec2i;

//wrapper class for Triangle library
class Delaunay{
  private:
    struct triangulateio out;
    std::vector<Triangle> triangles_;
    std::vector<Triangle> neighbours_;
    std::vector<Edge> edges_;

  public:
    void triangulate(const std::vector<DepthFeaturePair> &sparse_supports, std::vector<Triangle> & triangles);
    void cleanup();
    void getTriangles(std::vector<Triangle>* triangles);
    void getNeighbors();
    void getEdges();
    void drawWireframe(const Params & params, const std::vector<DepthFeaturePair> &sparse_supports, const std::vector<Triangle> & triangles, cv::Mat & img);

    // Accessors.
  const std::vector<Triangle>& triangles() const { return triangles_; }
  const std::vector<Edge>& edges() const { return edges_; }
  const std::vector<Triangle>& neighbors() const { return neighbours_; }
};





}

#endif
