#include "draw_utils.h"

/* ADAPTED FROM FLAME */

inline int min3(int x, int y, int z) {
  return x < y ? (x < z ? x : z) : (y < z ? y : z);
}

inline int max3(int x, int y, int z) {
  return x > y ? (x > z ? x : z) : (y > z ? y : z);
}

struct Edge {
  static const int stepXSize = 4;
  static const int stepYSize = 1;

  // __m128 is the SSE 128-bit packed float type (4 floats).
  __m128 oneStepX;
  __m128 oneStepY;

  __m128 init(const cv::Point& v0, const cv::Point& v1,
              const cv::Point& origin) {
    // Edge setup
    float A = v1.y - v0.y;
    float B = v0.x - v1.x;
    float C = v1.x*v0.y - v0.x*v1.y;

    // Step deltas
    // __m128i y = _mm_set1_ps(x) sets y[0..3] = x.
    oneStepX = _mm_set1_ps(A*stepXSize);
    oneStepY = _mm_set1_ps(B*stepYSize);

    // x/y values for initial pixel block
    // NOTE: Set operations have arguments in reverse order!
    // __m128 y = _mm_set_epi32(x3, x2, x1, x0) sets y0 = x0, etc.
    __m128 x = _mm_set_ps(origin.x + 3, origin.x + 2, origin.x + 1, origin.x);
    __m128 y = _mm_set1_ps(origin.y);

    // Edge function values at origin
    // A*x + B*y + C.
    __m128 A4 = _mm_set1_ps(A);
    __m128 B4 = _mm_set1_ps(B);
    __m128 C4 = _mm_set1_ps(C);

    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(A4, x), _mm_mul_ps(B4, y)), C4);
  }
};

// NB image must be type CV32FC1
void sandbox::drawShadedTriangleBarycentric(cv::Point p1, cv::Point p2, cv::Point p3,
                                   float v1, float v2, float v3, cv::Mat* img) {
  // Compute triangle bounding box
  int xmin = min3(p1.x, p2.x, p3.x);
  int ymin = min3(p1.y, p2.y, p3.y);
  int xmax = max3(p1.x, p2.x, p3.x);
  int ymax = max3(p1.y, p2.y, p3.y);

  cv::Point p(xmin, ymin);
  Edge e12, e23, e31;

  // __m128 is the SSE 128-bit packed float type (4 floats).
  __m128 w1_row = e23.init(p2, p3, p);
  __m128 w2_row = e31.init(p3, p1, p);
  __m128 w3_row = e12.init(p1, p2, p);

  // Values as 4 packed floats.
  __m128 v14 = _mm_set1_ps(v1);
  __m128 v24 = _mm_set1_ps(v2);
  __m128 v34 = _mm_set1_ps(v3);

  // Rasterize
  for (p.y = ymin; p.y <= ymax; p.y += Edge::stepYSize) {
    // Determine barycentric coordinates
    __m128 w1 = w1_row;
    __m128 w2 = w2_row;
    __m128 w3 = w3_row;

    for (p.x = xmin; p.x <= xmax; p.x += Edge::stepXSize) {
      // If p is on or inside all edges, render pixel.
      __m128 zero = _mm_set1_ps(0.0f);

      // (w1 >= 0) && (w2 >= 0) && (w3 >= 0)
      // mask tells whether we should set the pixel.
      __m128 mask = _mm_and_ps(_mm_cmpge_ps(w1, zero),
                               _mm_and_ps(_mm_cmpge_ps(w2, zero),
                                          _mm_cmpge_ps(w3, zero)));

      // w1 + w2 + w3
      __m128 norm = _mm_add_ps(w1, _mm_add_ps(w2, w3));

      // v1*w1 + v2*w2 + v3*w3 / norm
      __m128 vals = _mm_div_ps(_mm_add_ps(_mm_mul_ps(v14, w1),
                                          _mm_add_ps(_mm_mul_ps(v24, w2),
                                                     _mm_mul_ps(v34, w3))), norm);

      // Grab original data.  We need to use different store/load functions if
      // the address is not aligned to 16-bytes.
      uint32_t addr = sizeof(float)*(p.y*img->cols + p.x);
      if (addr % 16 == 0) {
        float* img_ptr = reinterpret_cast<float*>(&(img->data[addr]));
        __m128 data = _mm_load_ps(img_ptr);

        // Set values using mask.
        // If mask is true, use vals, otherwise use data.
        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
        _mm_store_ps(img_ptr, res);
      } else {
        // Address is not 16-byte aligned. Need to use special functions to load/store.
        float* img_ptr = reinterpret_cast<float*>(&(img->data[addr]));
        __m128 data = _mm_loadu_ps(img_ptr);

        // Set values using mask.
        // If mask is true, use vals, otherwise use data.
        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
        _mm_storeu_ps(img_ptr, res);
      }

      // One step to the right.
      w1 = _mm_add_ps(w1, e23.oneStepX);
      w2 = _mm_add_ps(w2, e31.oneStepX);
      w3 = _mm_add_ps(w3, e12.oneStepX);
    }

    // Row step.
    w1_row = _mm_add_ps(w1_row, e23.oneStepY);
    w2_row = _mm_add_ps(w2_row, e31.oneStepY);
    w3_row = _mm_add_ps(w3_row, e12.oneStepY);
  }

  return;
}


void sandbox::interpolateMesh(const std::vector<Triangle>& triangles, const std::vector<sandbox::DepthFeaturePair> & support_points, cv::Mat & dense_map){

  //cv::Mat dense_map_16bit(dense_map.size().height, dense_map.size().width, CV_16SC1);
  //dense_map.convertTo(dense_map_16bit,CV_16SC1,65536.0/255.0);

  //std::cout<<"type: "<<dense_map_16bit.type() << std::endl;

  for(auto t : triangles){
    cv::Point2f vert1 = support_points[t[0]].kp1.pt;
    cv::Point2f vert2 = support_points[t[1]].kp1.pt;
    cv::Point2f vert3 = support_points[t[2]].kp1.pt;
    float depth1 = support_points[t[0]].depth;
    float depth2 = support_points[t[1]].depth;
    float depth3 = support_points[t[2]].depth;

    //points are expected CCW, NB image must be type CV32FC1
    drawShadedTriangleBarycentric(vert3,vert2,vert1,depth3,depth2,depth1,&dense_map);

    
  }


}
