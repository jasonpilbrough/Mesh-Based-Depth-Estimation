#include "stereo_matching.h"
#include "colourmap.h"

using namespace cv;
using namespace xfeatures2d;
using namespace boost::filesystem;



void sandbox::StereoMatching::block_matching(const Params & params, const cv::Mat & img1, const cv::Mat & img2, cv::Mat & disp_img){

  cv::Mat img1_gray, img2_gray;
  cvtColor(img1, img1_gray, cv::COLOR_RGB2GRAY);
  cvtColor(img2, img2_gray, cv::COLOR_RGB2GRAY);

  int numDisparities=params.NUM_DISPARITIES; //must be multiple of 16 - default 64
  int blockSize=params.BLOCK_SIZE; //should be odd number between 5..11 - default 3

  cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(numDisparities, blockSize);
  matcher->compute(img1_gray, img2_gray, disp_img);

  disp_img.convertTo(disp_img, CV_32FC1);
  disp_img = disp_img /16.0f;

}



void sandbox::StereoMatching::semi_global_block_matching(const Params & params, const cv::Mat & img1, const cv::Mat & img2, cv::Mat & disp_img){

  cv::Mat img1_gray, img2_gray;
  cvtColor(img1, img1_gray, cv::COLOR_RGB2GRAY);
  cvtColor(img2, img2_gray, cv::COLOR_RGB2GRAY);

  int minDisparity=0;
  int numDisparities=params.NUM_DISPARITIES; //must be multiple of 16 - default 16
  int blockSize=params.BLOCK_SIZE; //should be odd number between 1..11 - default 3
  int P1=8*3*blockSize*blockSize;
  int P2=32*3*blockSize*blockSize;
  int disp12MaxDiff=1; //default 0
  int preFilterCap=0;
  int uniquenessRatio=0; // should be between 5..15 - default 0
  int speckleWindowSize=200; //should be between 50-200 - default 0
  int speckleRange=2; //should be between 1..2 - default 0
  int mode=cv::StereoSGBM::MODE_SGBM_3WAY; //default - MODE_SGBM

  cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2,
    disp12MaxDiff,preFilterCap,uniquenessRatio,speckleWindowSize, speckleRange,mode);

  matcher->compute(img1_gray, img2_gray, disp_img);

  disp_img.convertTo(disp_img, CV_32FC1);
  disp_img = disp_img /16.0f;

}



//generate sparse depth map from disparity values sampled from a dense disparity map
void sandbox::StereoMatching::sparse_matching(const Params & params, const Mat & img_l, const Mat & img_r, const Mat & disp_img, const float f, const float b, std::vector<DepthFeaturePair> & support_points, cv::Mat & support_points_img){

    //std::cout << disp_img.type() << std::endl;
    int width = img_l.size().width;
    int height = img_l.size().height;

    //std::vector<sandbox::DepthFeaturePair> support_points;

    //extract features from ref image
    int minHessian = params.MIN_HESSIAN;
    std::vector<cv::KeyPoint> keypoints1;
    cv::Ptr<cv::Feature2D> detector = cv::FastFeatureDetector::create( minHessian);
    detector->detect( img_l, keypoints1, cv::noArray());

    /*
    cv::Mat temp_img = support_points_img.clone();
    for (int i = 0; i < keypoints1.size();  i++){
      cv::Point2f vert = keypoints1[i].pt;
      cv::Vec3b c1(89,89,234);
      circle(temp_img, vert,1, c1,FILLED);
    }
    imshow("Sparse Keypoints",temp_img);
    */

    //SAMPLE FEATURES FROM SPACIAL BINS
    int bin_size_x = params.GRID_SIZE; //16
    int bin_size_y = params.GRID_SIZE; //16
    int num_bins = ((width / bin_size_x)+1) * ((height / bin_size_y)+1);
    std::vector<cv::KeyPoint> binned_keypoints1(num_bins);

    for(auto it = keypoints1.begin(); it!= keypoints1.end(); it++){
      int x_bin = it->pt.x / bin_size_x;
      int y_bin = it->pt.y / bin_size_y;
      int index = y_bin*(width/bin_size_x)+x_bin;

      if(it->response > binned_keypoints1[index].response){
          binned_keypoints1[index] = *it;
      }
    }

    binned_keypoints1.erase(std::remove_if(
      binned_keypoints1.begin(), binned_keypoints1.end(), [](const cv::KeyPoint& kp) {
        return kp.response <= 0; // put your condition here
      }),binned_keypoints1.end()
    );

    keypoints1 = binned_keypoints1;

    /*
    cv::Mat temp_img2 = support_points_img.clone();
    for (int i = 0; i < keypoints1.size();  i++){
      cv::Point2f vert = keypoints1[i].pt;
      cv::Vec3b c1(89,89,234);
      circle(temp_img2, vert,2, c1,FILLED);
    }
    imshow("Sparse Keypoints2",temp_img2);
    */

    //cv::Mat img_l_grey, img_r_grey;
    //cv::cvtColor(img_l, img_l_grey, COLOR_BGR2GRAY);
    //cv::cvtColor(img_r, img_r_grey, COLOR_BGR2GRAY);
    //vector<int> disparities(keypoints1.size(),0);
    //ssd(img_l_grey, img_r_grey, keypoints1, disparities);


    for(auto i = 0; i < keypoints1.size(); i++){

      float x = keypoints1[i].pt.x;
      float y = keypoints1[i].pt.y;

      float disparity = (disp_img.at<float>(y,x));
      //float disparity = (float)disparities[i];

      float depth = f*b / (disparity);

      if(depth<0){ //ignore features that have a negative corresponding disparity
        continue;
      }

      //ignore features that are too close to the border of the image
      float img_pad = params.IMG_PADDING;
      if(x > (img_l.size().width-img_pad) || x<img_pad || y > img_l.size().height-img_pad || y < img_pad){
        continue;
      }

      cv::KeyPoint dummyKP = cv::KeyPoint(x-disparity,y,0);
      //DepthFeaturePair dp(keypoints1[i],dummyKP,depth); //second keypoint is just a dummy
      DepthFeaturePair dp(keypoints1[i],dummyKP,disparity); //second keypoint is just a dummy
      support_points.push_back(dp);

    }


    //cv::Mat d_img = img_l.clone();
    //Mat d_img(cv::Size(img_l.size().width, img_l.size().height), CV_8UC3, Scalar(0));
    draw_depthmap(params, support_points, support_points_img);

}


void sandbox::StereoMatching::draw_depthmap(const Params & params, const std::vector<DepthFeaturePair> & support_points, cv::Mat & d_img){

  /*
  auto fiveNum = fiveNumSummary(support_points);
  float min = fiveNum[0];
  float percent5 =fiveNum[1];
  float Q1 = fiveNum[2];
  float Q2 = fiveNum[3];
  float Q3 = fiveNum[4];
  float percent95 = fiveNum[5];
  float max = fiveNum[6];
  */

  cv::Mat depths(1,support_points.size(), CV_32FC1);
  cv::Mat depths_norm(1,support_points.size(), CV_32FC1);

  std::transform( support_points.begin(), support_points.end(), depths.begin<float>(),
                [](DepthFeaturePair const& p) -> float { return p.depth; } );

  //cv::normalize(depths,depths_norm,0, 255, cv::NORM_MINMAX, -1);
  depths = (depths / (float)params.NUM_DISPARITIES*params.COLOUR_SCALE) * 255.0f;
  std::transform(depths.begin<float>(), depths.end<float>(), depths_norm.begin<float>(),
             [](float f) -> float { return std::max(0.0f, std::min(f, 255.0f)); }); //clamp between 0 and 255




  float radius = 2;

  for (int i = 0; i < support_points.size();  i++){

    cv::Point2f vert = support_points[i].kp1.pt;
    int depth_norm = static_cast<int>(depths_norm.at<float>(0,i));
    cv::Vec3b c1;
    COLOUR_MAP.lookup(depth_norm, c1);


    circle(d_img, vert,radius, c1,FILLED);

  }

}
