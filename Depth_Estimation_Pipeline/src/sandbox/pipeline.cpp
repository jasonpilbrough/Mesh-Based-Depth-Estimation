#include "pipeline.h"
#include "params.h"
#include "framemanager.h"
#include "stereo_matching.h"
#include "delaunay.h"
#include "colourmap.h"
#include "timing.h"
#include "draw_utils.h"
#include "regularisers.h"


sandbox::Pipeline::Pipeline(const std::string & stereo_l_filepath, const std::string & stereo_r_filepath) :
  fm(stereo_l_filepath,stereo_r_filepath) {}

sandbox::Pipeline::Pipeline(const std::string & stereo_l_filepath, const std::string & stereo_r_filepath, const std::string & stereo_gnd_filepath) :
  fm(stereo_l_filepath,stereo_r_filepath,stereo_gnd_filepath) {}


  void sandbox::Pipeline::run(){

    Timer t;
    Params params;
    Evaluation_stats eva_stats;

    StereoMatching stereo_matching;
    Delaunay delaunay;
    Mesh_Regulariser regulariser;

    float f = 1; //dummy value
    float b = 1; //dummy value


    auto start = std::chrono::system_clock::now();
    cv::Mat img_l, img_r, img_gnd;

    while(fm.hasNext()){



        //fm.nextFrame(img_l, img_r);

        ///*
        bool done = fm.nextFrame(img_l, img_r,img_gnd);
        if(done){
          float avg_MSE = (float)eva_stats.total_MSE / (float) eva_stats.img_count;
          float avg_MAE_rel = (float)eva_stats.total_MAE_Rel / (float) eva_stats.img_count;

          float average_delta10 = (float)eva_stats.count_delta10/ (float) eva_stats.img_count;
          float average_delta5 = (float)eva_stats.count_delta5/ (float) eva_stats.img_count;
          float average_delta1 = (float)eva_stats.count_delta1/ (float) eva_stats.img_count;

          auto end = std::chrono::system_clock::now();
          std::chrono::duration<double> elapsed_seconds = end-start;
          float avg_fps = (float) eva_stats.img_count/elapsed_seconds.count();

          std::cout <<"Evaluation Complete: "<<std::endl;
          std::cout <<"Num images: "<< eva_stats.img_count<<std::endl;
          std::cout <<"MSE: "<< avg_MSE<<std::endl;
          std::cout <<"MAE_Rel: "<< avg_MAE_rel<<std::endl;
          std::cout <<"Delta10: "<< average_delta10<<std::endl;
          std::cout <<"Delta5: "<< average_delta5<<std::endl;
          std::cout <<"Delta1: "<< average_delta1<<std::endl;
          std::cout <<"Avg fps: "<< avg_fps<<std::endl;

          break;
        }
        //*/


        /* ---------------------- STAGE 1: STEREO MATCHING  ---------------------- */
        std::vector<DepthFeaturePair> support_points;
        cv::Mat support_points_img = img_l.clone();
        //cv::Mat support_points_img(cv::Mat(img_l.size().height, img_l.size().width, CV_8UC3, cv::Scalar(0)));

        cv::Mat initial_disp_img, disp_img_eval, disp_img_show;
        stereo_matching.block_matching(params, img_l, img_r, initial_disp_img);
        //stereo_matching.semi_global_block_matching(params, img_l, img_r, initial_disp_img);
        stereo_matching.sparse_matching(params, img_l, img_r, initial_disp_img, f, b, support_points, support_points_img);
        t.tick();


        initial_disp_img.convertTo(disp_img_eval, CV_32FC1);
        //disp_img_eval = disp_img_eval / 16.0f;
        disp_img_show = initial_disp_img.clone();

        disp_img_show = (disp_img_show / (float)params.NUM_DISPARITIES*params.COLOUR_SCALE) * 255.0f;
        std::transform(disp_img_show.begin<float>(), disp_img_show.end<float>(), disp_img_show.begin<float>(),
                   [&params](float f) -> float { return std::max(0.0f, std::min(f, 255.0f)); }); //clamp between 0 and 255
        disp_img_show.convertTo(disp_img_show, CV_8U);
        COLOUR_MAP.applyColourMap(disp_img_show, disp_img_show);


        /* ------------------ STAGE 2: DELAUNAY TRIANGULATION  ------------------ */
        std::vector<Triangle> triangles;
        delaunay.triangulate(support_points,triangles);

        cv::Mat delaunay_unsmooth_img=img_l.clone();
        delaunay.drawWireframe(params, support_points,triangles,delaunay_unsmooth_img);

        //cv::Mat disp_dense_unsmooth, disp_dense_unsmooth_show;
        //disp_dense_unsmooth = cv::Mat(img_l.size().height, img_l.size().width, CV_32FC1, cv::Scalar(0));
        //interpolateMesh(triangles,support_points,disp_dense_unsmooth);
        //disp_dense_unsmooth = (disp_dense_unsmooth / (float)params.NUM_DISPARITIES*params.COLOUR_SCALE) * 255.0f;
        //std::transform(disp_dense_unsmooth.begin<float>(), disp_dense_unsmooth.end<float>(), disp_dense_unsmooth.begin<float>(),
        //           [](float f) -> float { return std::max(0.0f, std::min(f, 255.0f)); }); //clamp between 0 and 255

        //disp_dense_unsmooth.convertTo(disp_dense_unsmooth_show, CV_8U);
        //COLOUR_MAP.applyColourMap(disp_dense_unsmooth_show, disp_dense_unsmooth_show);



        /* ------------------ STAGE 3: VARIATIONAL SMOOTHING  ------------------ */
        std::vector<DepthFeaturePair> support_points_reg = support_points;
        std::vector<Edge> edges = delaunay.edges();


        //regulariser.run_TV(support_points,edges,support_points_reg);
        //regulariser.run_TGV(support_points,edges,support_points_reg);
        regulariser.run_logTV(support_points,edges,support_points_reg);
        //regulariser.run_logTGV(support_points,edges,support_points_reg);



        cv::Mat delaunay_smooth_img=img_l.clone();
        delaunay.drawWireframe(params, support_points_reg,triangles,delaunay_smooth_img);

        cv::Mat disp_dense_smooth, disp_dense_smooth_eval, disp_dense_smooth_show;
        disp_dense_smooth = cv::Mat(img_l.size().height, img_l.size().width, CV_32FC1, cv::Scalar(0));

        interpolateMesh(triangles,support_points_reg, disp_dense_smooth);

        disp_dense_smooth.convertTo(disp_dense_smooth_eval, CV_32FC1); //just make a copy
        disp_dense_smooth_show = (disp_dense_smooth / (float)params.NUM_DISPARITIES*params.COLOUR_SCALE) * 255.0f;
        std::transform(disp_dense_smooth_show.begin<float>(), disp_dense_smooth_show.end<float>(), disp_dense_smooth_show.begin<float>(),
                   [&params](float f) -> float { return std::max(0.0f, std::min(f, 255.0f)); }); //clamp between 0 and 255
        disp_dense_smooth_show.convertTo(disp_dense_smooth_show, CV_8U);
        COLOUR_MAP.applyColourMap(disp_dense_smooth_show, disp_dense_smooth_show);


        t.tock();

        /* ----------------------- PROCESS GROUND TRUTH  ----------------------- */
        cv::Mat img_gnd_show;
        img_gnd.convertTo(img_gnd, CV_8UC1);
        cv::cvtColor(img_gnd, img_gnd, cv::COLOR_BGR2GRAY);

        std::transform(img_gnd.begin<char>(), img_gnd.end<char>(), img_gnd.begin<char>(),
                   [&params](char f) -> char { return (char)std::max(0, std::min((int)f, params.NUM_DISPARITIES-1)); }); //clamp between 0 and 255
        img_gnd_show = img_gnd * (256.0/(float)params.NUM_DISPARITIES*params.COLOUR_SCALE);
        img_gnd_show.convertTo(img_gnd_show, CV_8U);
        COLOUR_MAP.applyColourMap(img_gnd_show, img_gnd_show);

        //evaluate(img_gnd,disp_img_eval,eva_stats);
        evaluate(img_gnd,disp_dense_smooth_eval,eva_stats);



        t.draw_timing(delaunay_smooth_img);

        //imshow("Original Image", img_l);
        //imshow("Disparity Map", disp_img_show);
        //imshow("Unsmoothed Depth Map", disp_dense_unsmooth_show);
        //imshow("Unsmoothed Delaunay", delaunay_unsmooth_img );
        imshow("Smoothed Delaunay", delaunay_smooth_img );
        //imshow("Features", support_points_img);
        //imshow("Ground Truth", img_gnd_show );
        //imshow("Smoothed Depth Map", disp_dense_smooth_show);

        //Opencv triggers the actual drawing in waitKey(), so the delay is likely to take at least 10-20ms
        char c=(char)cv::waitKey(1);
        if(c==113|| c==32){
          break;
        }




        cv::waitKey(0);

    }
  }


  void sandbox::Pipeline::evaluate(const cv::Mat & img_gnd, const cv::Mat & img_guess, Evaluation_stats & stats){

    if(img_gnd.type()!=0){
      std::cout << "Warning unsupported type for img_gnd in evaluation -> type="<<img_gnd.type() << std::endl;
    }
    if(img_guess.type()!=5){
      std::cout << "Warning unsupported type for img_guess in evaluation -> type="<<img_guess.type()  << std::endl;
    }

    //std::cout << img_gnd.type()<<" : "<< img_guess.type() << std::endl;
    int count = 0;
    int sse = 0;
    float sum_RE = 0; //relative inverse depth error

    int count_delta10 = 0; //density of accurate depth estimates
    int count_delta5 = 0;
    int count_delta1 = 0;

    for(int i=0; i<img_gnd.rows; i++){
      for(int j=0; j<img_gnd.cols; j++){
        int gnd_val = (int)img_gnd.at<char>(i,j);
        int guess_val = std::round(img_guess.at<float>(i,j));

        //dont count pixels with no ground truth
        if(gnd_val==0||guess_val==0){
          continue;
        }

        //if(i>200 && i<300 && j>700 && j<800){
            //std::cout << gnd_val<<" : "<< guess_val<< std::endl;
        //}


        float diff = std::abs(gnd_val-guess_val);

        //if(diff>1){
        //  std::cout << diff << " : " << gnd_val*0.1<< std::endl;
        //}

        sse+=diff*diff;
        sum_RE += ((float)diff/ (float)gnd_val);

        if(diff<gnd_val*0.1){
            count_delta10++;
        }
        if(diff<gnd_val*0.05){
            count_delta5++;
        }
        if(diff<gnd_val*0.01){
            count_delta1++;
        }

        count++;
      }
    }
    float MSE = (float)sse / (float)count;
    float MAE_Rel = sum_RE / (float)count;

    float average_delta10 = (float)count_delta10 /(float) count;
    float average_delta5 = (float)count_delta5 /(float) count;
    float average_delta1 = (float)count_delta1 /(float) count;

    stats.img_count++;
    stats.total_MSE += MSE;
    stats.total_MAE_Rel += MAE_Rel;

    stats.count_delta10 += average_delta10;
    stats.count_delta5 += average_delta5;
    stats.count_delta1 += average_delta1;

    //std::cout <<"Img index: "<< stats.img_count << "Avg. Error: "<< avg_error << "   Relative Error[%]: " << average_RE << "  Error <10%[%]: "<< average_AD << " Num pxs: "<< count << " out of "<< img_gnd.size().height * img_gnd.size().width << std::endl;
    std::cout <<"IND="<< stats.img_count << " MSE="<<MSE <<" MAE_Rel=" << MAE_Rel << "  D10="<< average_delta10 << "  D5="<< average_delta5 << "  D1="<< average_delta1 << std::endl;

  }



  void sandbox::Pipeline::loadCalib(cv::Mat & K1, cv::Mat & K2, cv::Mat & T1, cv::Mat & T2){

    K1= (cv::Mat_<float>(3,3) <<   984.242, 0.00, 690.00, 0.00, 980.81, 233.196, 0.00, 0.00, 1.00);
    K2 = (cv::Mat_<float>(3,3) <<   989.526, 0.00, 702.00, 0.000, 987.838, 245.55, 0.00, 0.00, 1.00);

    T1= (cv::Mat_<float>(3,1) <<   0,0,0);
    T2 = (cv::Mat_<float>(3,1) <<   -0.537, 0.00482, -0.0125);

    /*
    std::ifstream fstream("data/KITTI/calib/calib_cam_to_cam.txt");
    std::map<std::string, cv::Mat*> calibLUT;


    std::string line, key, data;

    //ignore first two lines
    getline(fstream,line);
    getline(fstream,line);

    while(getline(fstream,line)){

      std::vector<std::string> splitKeyData, splitData;
      std::vector<double> values;

      boost::algorithm::split(splitKeyData, line, boost::is_any_of(":"));
      key = splitKeyData[0];
      data = splitKeyData[1];

      //trim whitespace
      boost::trim_left(data);
      boost::trim_right(data);

      boost::algorithm::split(splitData, data, boost::is_any_of(" "));

      for(int i = 0; i<splitData.size(); i++){
         std::string::size_type sz;     // alias of size_t
         double val = std::stod(splitData[i],&sz);
         values.push_back(val);
      }

      cv::Mat * matrix = new cv::Mat(1, values.size(), CV_64F, values.data());
      //calibLUT[key] = matrix;
      calibLUT.insert(std::make_pair<std::string ,cv::Mat*>(std::move(key), std::move(matrix)));

      std::cout << key <<" "<< *calibLUT["S_00"] << std::endl;

    }

    std::cout <<  *calibLUT["S_00"] << std::endl;
    */

  }
