#include "framemanager.h"
#include "colourmap.h"


sandbox::FrameManager::FrameManager(const std::string & path_l, const std::string & path_r)
          : dataset_path_left(path_l), dataset_path_right(path_r){

      cap_left =  cv::VideoCapture(path_l);
      cap_right = cv::VideoCapture(path_r);

      if(!cap_left.isOpened() || !cap_right.isOpened()){
        std::cout << "Error opening video stream (left or right)" << std::endl;
        return;
      }

    };

sandbox::FrameManager::FrameManager(const std::string & path_l, const std::string & path_r,const std::string & path_gnd)
          : dataset_path_left(path_l), dataset_path_right(path_r), dataset_path_gnd(path_gnd){

      cap_left =  cv::VideoCapture(path_l);
      cap_right = cv::VideoCapture(path_r);
      cap_gnd = cv::VideoCapture(path_gnd);

      if(!cap_left.isOpened() || !cap_right.isOpened()){
        std::cout << "Error opening video stream (left or right)" << std::endl;
        return;
      }

      if(!cap_gnd.isOpened()){
        std::cout << "Error opening video stream (gnd truth)" << std::endl;
        return;
      }

    };


bool sandbox::FrameManager::hasNext(){
  return cap_left.isOpened() && cap_right.isOpened();
}

bool sandbox::FrameManager::nextFrame(cv::Mat & img_l, cv::Mat & img_r,cv::Mat & img_gnd){

      if(!cap_left.isOpened() || !cap_right.isOpened() || !cap_gnd.isOpened()){
        std::cout << "Error retrieving video or ground truth" << std::endl;
        return true;
      }
      cv::Mat dummy;
      //skip every second image when no ground truth provided
      cap_left.read(img_l);
      //cap_left.read(dummy);
      cap_right.read(img_r);
      //cap_right.read(dummy);
      cap_gnd.read(img_gnd);


      if (img_l.empty()||img_r.empty()||img_gnd.empty()){
        return true;
      }

      // CROP IMAGE
      //cv::Rect myROI(230, 0, img_l.size().width-(2*230), img_l.size().height);
      //img_l = img_l(myROI);
      //img_r = img_r(myROI);
      //img_gnd = img_gnd(myROI);

      return false;
}


void sandbox::FrameManager::nextFrame(cv::Mat & img_l, cv::Mat & img_r){

      if(!cap_left.isOpened() || !cap_right.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return;
      }

      cap_left.read(img_l);
      cap_right.read(img_r);


      if (img_l.empty()||img_r.empty()){
        //std::cout << "End of sequence - restarting" << std::endl;
        cap_left =  cv::VideoCapture(dataset_path_left);
        cap_right = cv::VideoCapture(dataset_path_right);
        cap_left.read(img_l);
        cap_right.read(img_r);
      }

      //cv::imshow("Img left", img_l);
      //cv::imshow("Img right", img_r);
      //cv::waitKey(0);


      /*
      //EuRoC
      //Parameters from ORB-SLAM
      cv::Mat K1 = (cv::Mat_<double>(3,3)  << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
      cv::Mat R1 = (cv::Mat_<double>(3,3)  << 0.999966347530033, -0.001422739138722922, 0.008079580483432283, 0.001365741834644127, 0.9999741760894847, 0.007055629199258132, -0.008089410156878961, -0.007044357138835809, 0.9999424675829176);
      cv::Mat P1 = (cv::Mat_<double>(3,4)  << 435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0);
      cv::Mat d1 = (cv::Mat_<double>(1,5)  <<  -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0);

      cv::Mat K2 = (cv::Mat_<double>(3,3)  << 457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1);
      cv::Mat R2 = (cv::Mat_<double>(3,3)  << 0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 0.003680398547259526, 0.9999684752771629, -0.007035845251224894, -0.007729688520722713, 0.007064130529506649, 0.999945173484644);
      cv::Mat P2 = (cv::Mat_<double>(3,4)  << 435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0);
      cv::Mat d2 = (cv::Mat_<double>(1,5)  <<  -0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0);

      int rows_l = 480;
      int cols_l = 752;
      int rows_r = 480;
      int cols_r = 752;


      cv::Mat M1l,M2l,M1r,M2r;
      cv::initUndistortRectifyMap(K1,d1,R1,P1.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
      cv::initUndistortRectifyMap(K2,d2,R2,P2.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);

      cv::Mat img_l_new,img_r_new;
      cv::remap(img_l,img_l,M1l,M2l,cv::INTER_LINEAR);
      cv::remap(img_r,img_r,M1r,M2r,cv::INTER_LINEAR);
      */

      //cv::imshow("Img old", img_l);
      //cv::imshow("Img new", img_l_new);


      //t1 = R1.inv()*t1;
      //t1 = R2.inv()*t2;

      /*
      //KITTI
      cv::Mat K1= (cv::Mat_<double>(3,3) <<   9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00);
      cv::Mat K2 = (cv::Mat_<double>(3,3) <<   9.895267e+02, 0.000000e+00, 7.020000e+02, 0.000000e+00, 9.878386e+02, 2.455590e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00);
      cv::Mat R1= (cv::Mat_<double>(3,3) <<   1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00);
      cv::Mat R2= (cv::Mat_<double>(3,3) <<   9.993513e-01, 1.860866e-02, -3.083487e-02, -1.887662e-02, 9.997863e-01, -8.421873e-03, 3.067156e-02, 8.998467e-03, 9.994890e-01);
      cv::Mat t1= (cv::Mat_<double>(3,1) <<   2.573699e-16, -1.059758e-16, 1.614870e-16);
      cv::Mat t2 = (cv::Mat_<double>(3,1) <<  -5.370000e-01, 4.822061e-03, -1.252488e-02);
      cv::Mat d1= (cv::Mat_<double>(1,5) << -3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02);
      cv::Mat d2= (cv::Mat_<double>(1,5) << -3.644661e-01, 1.790019e-01, 1.148107e-03, -6.298563e-04, -5.314062e-02);
      */

      /*
      //cv::Mat R12 = R1.t() * R2;
      //cv::Mat t12 = R1.t() * t2 + R1.t() * t1;
      //cv::Mat R21 = R2.t() * R1;
      //cv::Mat t21 = R2.t() * t1 + R2.t() * t2;

      cv::Mat R12 = R1.t() * R2; //inverse of rotation matrix is transpose
      cv::Mat t12 = R1.t() * t2 + R1.t() * t1;
      cv::Mat R21 = R2.t() * R1;
      cv::Mat t21 = R2.t() * t1 + R2.t() * t2;

      cv::Mat R1_rec, R2_rec, P1_rec, P2_rec, Q;
      //cv::Size newSize(1242, 375);
      cv::Size newSize(1200,700);
      cv::stereoRectify(K1, d1, K2, d2, img_l.size(), R12, t12, R1_rec, R2_rec, P1_rec, P2_rec, Q,cv::CALIB_ZERO_DISPARITY,-1, newSize);


      cv::Mat map11, map12, map21, map22;
      cv::initUndistortRectifyMap(K1, d1, R1_rec, P1_rec, newSize,CV_8UC1, map11, map12);
      cv::initUndistortRectifyMap(K2, d2, R2_rec, P2_rec, newSize,CV_8UC1, map21, map22); //R12
      //cv::initUndistortRectifyMap(K1, d1, R21, K1_new, img_l.size(),CV_8UC1, img_l_map1, img_l_map2);
      //cv::initUndistortRectifyMap(K2, d2, R12, K2_new, img_r.size(),CV_8UC1, img_r_map1, img_r_map2);

      cv::Mat img_l_new,img_r_new;
      cv::remap(img_l, img_l, map11, map12, cv::INTER_LINEAR);
      cv::remap(img_r, img_r, map21, map22, cv::INTER_LINEAR);

      //cv::imshow("Img old", img_r);
      //cv::imshow("Img new", img_r_new);



      */


      /*
      if(!cap_left.isOpened() || !cap_right.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return;
      }

      while(cap_left.isOpened() && cap_right.isOpened()){
        std::cout << "Here1" << std::endl;
        cap_left.read(img_l);
        cap_right.read(img_r);

        // If the frame is empty, break immediately
        if (img_l.empty()||img_r.empty())
          break;

        std::cout << "Here2" << std::endl;
        cv::imshow("frame1", img_l);
        cv::waitKey(0);

      }
      */

}



/*
FrameManager::FrameManager(const std::string & path_l, const std::string & path_r, bool seq) :
    filepath_left(path_l), filepath_right(path_r), sequence(seq){

      if(sequence){
        fstream_l.open(path_l);
        fstream_r.open(path_r);

        //remove header line from file stream
        std::string line_l = "";
        std::string line_r = "";
        getline(fstream_l, line_l);
        getline(fstream_r, line_r);
      }

    };

int FrameManager::loadFrame(const std::string & img1_filepath, const std::string & img2_filepath, cv::Mat & img1, cv::Mat & img2){

  cv::Mat img1_distorted = imread(img1_filepath, cv::IMREAD_COLOR); //IMREAD_GRAYSCALE
  cv::Mat img2_distorted = imread(img2_filepath, cv::IMREAD_COLOR);


  if ( img1_distorted.empty() || img2_distorted.empty() ){
      std::cout << "Could not open or find one of the images!" << std::endl;
      return -1;
  }


  cv::Mat K1 = (cv::Mat_<float>(3,3) <<   458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);
  cv::Mat K2 = (cv::Mat_<float>(3,3) <<   457.587, 0, 379.999, 0, 456.134, 255.238, 0, 0, 1);

  cv::Mat d_coeff1 = (cv::Mat_<float>(4,1) <<   -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
  cv::Mat d_coeff2 = (cv::Mat_<float>(4,1) <<   -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05);

  //cv::Mat P1 = (cv::Mat_<float>(3,4) <<   435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0);
  //cv::Mat P2 = (cv::Mat_<float>(3,4) <<   435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0);

  //cv::Mat K1, R1, T1, K2, R2, T2;

  //decomposeProjectionMatrix(P1,K1,R1,T1);
  //decomposeProjectionMatrix(P2,K2,R2,T2);

  //cv::Mat map1, map2;
  //cv::initUndistortRectifyMap(K1, d_coeff1, R1, K2,img1_distorted.size(),CV_32FC1, map1, map2 );

  //std::cout << map1.size() << std::endl;
  //remap(img1_distorted, img1, map1, map2, cv::INTER_LINEAR);
  //remap(img2_distorted, img2, map1, map2, cv::INTER_LINEAR);


  cv::Size img1_size_new, img2_size_new;

  cv::Mat K1_new = cv::getOptimalNewCameraMatrix(K1, d_coeff1, img1_distorted.size(), 0,img1_size_new);
  cv::Mat K2_new = cv::getOptimalNewCameraMatrix(K2, d_coeff2, img2_distorted.size(), 0,img2_size_new);

  img1 = img1_distorted;
  img2 = img2_distorted;

  //cv::undistort(img1_distorted, img1, K1, d_coeff1); //K1_new
  //cv::undistort(img2_distorted, img2, K2, d_coeff2); //K2_new


  //cv::imshow("Test1",img1);
  //cv::waitKey(0);
  //cv::imshow("Test2",img2);
  //cv::waitKey(0);

  return 0;
}

bool FrameManager::hasNext(){
  if(!sequence && !runOnce){
    runOnce = true;
    return true;
  }else if(!sequence && runOnce){
    return false;
  }else if(fstream_l && fstream_l.peek()== EOF && fstream_r && fstream_l.peek()== EOF){
    return false;
  }else{
    return true;
  }
}

void FrameManager::nextFrame(cv::Mat & img_l, cv::Mat & img_r){

  if(!sequence){
    loadFrame(filepath_left, filepath_right, img_l, img_r);
  }else{

    std::vector<std::string> vec_l, vec_r;
    std::string line_l, line_r;

    getline(fstream_l, line_l);
    getline(fstream_r, line_r);
    boost::algorithm::split(vec_l, line_l, boost::is_any_of(","));
    boost::algorithm::split(vec_r, line_r, boost::is_any_of(","));

    std::string  img_l_path = "data/mav0/cam0/data/"+vec_l[1].substr(0, vec_l[1].size()-1);
    std::string  img_r_path = "data/mav0/cam1/data/"+vec_r[1].substr(0, vec_r[1].size()-1);

    loadFrame(img_l_path, img_r_path, img_l, img_r);

    //file_l.close();
    //file_r.close();

  }

}

void FrameManager::nextFrame2(cv::Mat & img_l, cv::Mat & img_r){

  if(!sequence){
    loadFrame(filepath_left, filepath_right, img_l, img_r);
  }else{

    auto cap_left =  cv::VideoCapture("data/KITTI/dataset/2011_09_26_drive_0002_sync/image_00/data/0000000000.png");
    auto cap_right = cv::VideoCapture("data/KITTI/dataset/2011_09_26_drive_0002_sync/image_01/data/0000000000.png");

    if(!cap_left.isOpened() || !cap_right.isOpened()){
      std::cout << "Error opening video stream or file" << std::endl;
      return;
    }

    while(cap_left.isOpened() && cap_right.isOpened()){


      cap_left.read(img_l);
      cap_right.read(img_r);

      // If the frame is empty, break immediately
      if (img_l.empty()||img_r.empty())
        break;

    }

  }

}
*/

/*
void processVideo(float f, float b){

  std::ifstream file_l("data/mav0/cam0/data.csv");
  std::ifstream file_r("data/mav0/cam1/data.csv");

  std::string line_l = "";
  std::string line_r = "";
  getline(file_l, line_l); // read first header line
  getline(file_r, line_r); // read first header line

  Mat img1, img2;
  std::vector<std::string> vec_l, vec_r;
  string filepath_l,filepath_r;

  while (getline(file_l, line_l)&&getline(file_r, line_r)){

    auto start = std::chrono::system_clock::now();

    boost::algorithm::split(vec_l, line_l, boost::is_any_of(","));
    boost::algorithm::split(vec_r, line_r, boost::is_any_of(","));

    filepath_l = "data/mav0/cam0/data/"+vec_l[1].substr(0, vec_l[1].size()-1);
    filepath_r = "data/mav0/cam1/data/"+vec_r[1].substr(0, vec_r[1].size()-1);

    //filepath_l.erase(std::remove(filepath_l.begin(), filepath_l.end(), '\r'), filepath_l.end());
    //filepath_r.erase(std::remove(filepath_r.begin(), filepath_r.end(), '\r'), filepath_r.end());

    loadFrame(filepath_l, filepath_r, img1, img2);

    featureDepthEstimation(img1, img2, f, b);

    //cv::Mat win_mat(cv::Size(img1.size().width*2, img1.size().height), CV_8UC3);
    //img1.copyTo(win_mat(cv::Rect(  0, 0, img1.size().width, img1.size().height)));
    //img2.copyTo(win_mat(cv::Rect(img1.size().width, img1.size().height, img2.size().width, img2.size().height)));

    //cv::imshow("frame1", img1);

    // Press  'q on keyboard to exit
    //char c=(char)cv::waitKey(30);
    char c=(char)cv::waitKey(1);
    if(c==113|| c==32){
      break;
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "frame rate: " << 1/elapsed_seconds.count() <<std::endl;

  }

  file_l.close();
  file_r.close();


  auto cap = cv::VideoCapture("data/blais.mp4");

  if(!cap.isOpened()){
    std::cout << "Error opening video stream or file" << std::endl;
    return;
  }

  cv::Mat nextFrame;
  while(cap.isOpened()){

      auto start = std::chrono::system_clock::now();

    cap.read(nextFrame);

    // If the frame is empty, break immediately
    if (nextFrame.empty())
      break;

    cv::imshow("frame", nextFrame);

    // Press  'q on keyboard to exit
    char c=(char)cv::waitKey(1);
    if(c==113){
      break;
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "frame rate:" << 1/elapsed_seconds.count() <<std::endl;

  }


}
*/
