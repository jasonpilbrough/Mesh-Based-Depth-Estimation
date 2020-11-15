#include "pipeline.h"
//#include "colourmap.h"
//#include <opencv2/viz.hpp>



int main(int argc, char* argv[])
{


    // ######### EuRoC #########
    //std::string dataset_path_left  = "data/EuRoC/MH1/cam0/data/%10d.png";
    //std::string dataset_path_right = "data/EuRoC/MH1/cam1/data/%10d.png";

    // ######### ETH3D #########
    std::string dataset_path_left  = "data/ETH3D/delivery_area/cam4_brighter/%10d.png";
    std::string dataset_path_right = "data/ETH3D/delivery_area/cam5_brighter/%10d.png";
    std::string dataset_path_gnd   = "data/ETH3D/delivery_area/gnd/%10d.png";

    // ######### OXFORD ######### (This is actually MVSEC)
    //std::string dataset_path_left  = "data/Oxford/indoor_flying1/cam0/%10d.png";
    //std::string dataset_path_right = "data/Oxford/indoor_flying1/cam1/%10d.png";
    //std::string dataset_path_gnd   = "data/Oxford/indoor_flying1/gnd/%10d.png";

    sandbox::Pipeline p(dataset_path_left,dataset_path_right,dataset_path_gnd);
    p.run();



    return 0;
}



// ######### KITTI #########
//std::string dataset_path_left  = "data/KITTI/2011_09_26_drive_0002_sync/image_00/data/%10d.png";
//std::string dataset_path_right = "data/KITTI/2011_09_26_drive_0002_sync/image_01/data/%10d.png";
//std::string dataset_path_left  = "data/KITTI/data_stereo_flow/image_00/%10d.png";
//std::string dataset_path_right = "data/KITTI/data_stereo_flow/image_01/%10d.png";
//std::string dataset_path_gnd = "data/KITTI/data_stereo_flow/disp_noc/%10d.png";

//sandbox::Pipeline p(dataset_path_left,dataset_path_right);
