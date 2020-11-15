#ifndef _SANDBOX_TIMING
#define _SANDBOX_TIMING

#include <chrono>
#include <algorithm>
#include <numeric>

#include <boost/circular_buffer.hpp>
#include "boost/format.hpp"
#include <opencv2/core.hpp>

class Timer{
  private:
      std::chrono::time_point<std::chrono::system_clock> start;
      int cb_size = 50;
      boost::circular_buffer<float> cb;
      float count;

      float fps;
      float average_fps;
      float peak_fps;

  public:
      Timer(): cb(cb_size), count(0), peak_fps(0){}


      void tick(){
          start = std::chrono::system_clock::now();
      }
      void tock(){
          auto end = std::chrono::system_clock::now();
          std::chrono::duration<double> elapsed_seconds = end-start;

          fps = 1/elapsed_seconds.count();

          cb.push_back(fps);
          float sum = std::accumulate(cb.begin(), cb.end(), 0);
          average_fps = sum/(count < cb_size ? count++ : cb_size); //divide by count if count < (circular buffer size) else divide by (circular buffer size)

          //float peak_fps=0;
          //std::for_each(cb.begin(), cb.end(), [&peak_fps](float &n){ if(n>peak_fps) peak_fps=n;});

          if(fps > peak_fps){
            peak_fps = fps;
          }

          //boost::format fmt = boost::format("fps: %-6.2f avg_fps: %-6.2f peak_fps: %-6.2f") % fps % average_fps % peak_fps;
          //std::cout << fmt.str() << std::endl;
      }

    void draw_timing(cv::Mat & img){

        float average_fps_draw = average_fps;
        if(average_fps_draw<0){ //sometimes average is weird in the first couple of frames
          average_fps_draw=0;
        }

        boost::format fmt1 = boost::format("fps: %-6.2f") % fps;
        boost::format fmt2 = boost::format("avg_fps: %-6.2f") % average_fps_draw;
        boost::format fmt3 = boost::format("peak_fps: %-6.2f") % peak_fps;

        rectangle(img, cv::Point(2,6), cv::Point(68,18), cv::Scalar(0,0,0), cv::FILLED);
        cv::putText(img, fmt1.str(),
                  cv::Point(5,15), // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  0.5, // Scale. 2.0 = 2x bigger
                  cv::Scalar(255,255,255), // BGR Color
                  1 // Line Thickness (Optional)
                  //CV_AA // Anti-alias (Optional)
                );

        rectangle(img, cv::Point(2,20), cv::Point(100,32), cv::Scalar(0,0,0), cv::FILLED);
        cv::putText(img, fmt2.str(),
                  cv::Point(5,29), // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  0.5, // Scale. 2.0 = 2x bigger
                  cv::Scalar(255,255,255), // BGR Color
                  1 // Line Thickness (Optional)
                  //CV_AA // Anti-alias (Optional)
                );
        rectangle(img, cv::Point(2,34), cv::Point(105,46), cv::Scalar(0,0,0), cv::FILLED);
        cv::putText(img, fmt3.str(),
                  cv::Point(5,43), // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  0.5, // Scale. 2.0 = 2x bigger
                  cv::Scalar(255,255,255), // BGR Color
                  1 // Line Thickness (Optional)
                  //CV_AA // Anti-alias (Optional)
                );


    }

};




#endif
