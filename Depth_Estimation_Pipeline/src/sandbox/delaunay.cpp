#include "delaunay.h"
#include "colourmap.h"

/* ADAPTED FROM FLAME */

void sandbox::Delaunay::triangulate(const std::vector<DepthFeaturePair> & sparse_supports,
                                    std::vector<Triangle> & triangles
                                ){


  // input/output structure for triangulation
  struct triangulateio in;
  int32_t k;

  // inputs
  in.numberofpoints = sparse_supports.size();
  in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float)); // NOLINT
  k = 0;
  for (int32_t i = 0; i < sparse_supports.size(); i++) {
    in.pointlist[k++] = sparse_supports[i].kp1.pt.x;
    in.pointlist[k++] = sparse_supports[i].kp1.pt.y;
  }
  in.numberofpointattributes = 0;
  in.pointattributelist      = NULL;
  in.pointmarkerlist         = NULL;
  in.numberofsegments        = 0;
  in.numberofholes           = 0;
  in.numberofregions         = 0;
  in.regionlist              = NULL;

  // outputs
  out.pointlist              = NULL;
  out.pointattributelist     = NULL;
  out.pointmarkerlist        = NULL;
  out.trianglelist           = NULL;
  out.triangleattributelist  = NULL;
  out.neighborlist           = NULL;
  out.segmentlist            = NULL;
  out.segmentmarkerlist      = NULL;
  out.edgelist               = NULL;
  out.edgemarkerlist         = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
  char parameters[] = "zneQB";
  ::triangulate(parameters, &in, &out, NULL);
  free(in.pointlist);

  getTriangles(&triangles);
  getNeighbors();
  getEdges();
  cleanup();

  return;

}

void sandbox::Delaunay::cleanup() {
  // free memory used for triangulation
  free(out.pointlist);
  free(out.trianglelist);
  free(out.edgelist);
  free(out.neighborlist);

  out.pointlist = NULL;
  out.trianglelist = NULL;
  out.edgelist = NULL;
  out.neighborlist = NULL;

  return;
}

void sandbox::Delaunay::getTriangles(std::vector<Triangle>* triangles) {
  // put resulting triangles into vector tri
  triangles->resize(out.numberoftriangles);
  int k = 0;
  for (int32_t i = 0; i < out.numberoftriangles; i++) {
    (*triangles)[i] = Triangle(out.trianglelist[k],
                               out.trianglelist[k+1],
                               out.trianglelist[k+2]);
    k+=3;
  }
  return;
}

void sandbox::Delaunay::getNeighbors() {
  // put neighboring triangles into vector tri
  neighbours_.resize(out.numberoftriangles);
  int k = 0;
  for (int32_t i = 0; i < out.numberoftriangles; i++) {
    neighbours_[i] = Triangle(out.neighborlist[k],
                            out.neighborlist[k+1],
                            out.neighborlist[k+2]);
    k+=3;
  }
  return;
}

void sandbox::Delaunay::getEdges(){
  // put resulting edges into vector
  edges_.resize(out.numberofedges);
  int k = 0;
  for (int32_t i = 0; i < out.numberofedges; i++) {
    edges_[i] = Edge(out.edgelist[k], out.edgelist[k+1]);
    k+=2;
  }
  return;
}





void drawLineColourMap(cv::Mat& img, const cv::Point& start, const cv::Point& end, const int depth_norm1,   const int depth_norm2) {
    cv::Vec3b c1;
    cv::LineIterator iter(img, start, end, cv::LINE_8);


    for (int i = 0; i < iter.count; i++, iter++) {
         double alpha = double(i) / iter.count;

         sandbox::COLOUR_MAP.lookup(depth_norm1 * (1.0 - alpha) + depth_norm2 * alpha, c1);

         (*iter)[0] = (uint8_t)(c1[0]);
         (*iter)[1] = (uint8_t)(c1[1]);
         (*iter)[2] = (uint8_t)(c1[2]);
    }

    /*
    cv::LineIterator iter(img, start, end, cv::LINE_8);

    for (int i = 0; i < iter.count; i++, iter++) {
       double alpha = double(i) / iter.count;
       (*iter)[0] = (uint8_t)(c1[0] * (1.0 - alpha) + c2[0] * alpha);
       (*iter)[1] = (uint8_t)(c1[1] * (1.0 - alpha) + c2[1] * alpha);
       (*iter)[2] = (uint8_t)(c1[2] * (1.0 - alpha) + c2[2] * alpha);
    }
    */
}



void sandbox::Delaunay::drawWireframe(const Params & params, const std::vector<DepthFeaturePair> & sparse_supports, const std::vector<Triangle> & triangles, cv::Mat & img){

  int thickness = 1;
  int lineType = cv::LINE_8;

  //auto fiveNum = fiveNumSummary(sparse_supports);
  //float percent5 =fiveNum[1];
  //float percent95 = fiveNum[5];
  //float max_depth = fiveNum[6];


  cv::Mat depths(1,sparse_supports.size(), CV_32FC1);
  cv::Mat depths_norm(1,sparse_supports.size(), CV_32FC1);

  std::transform( sparse_supports.begin(), sparse_supports.end(), depths.begin<float>(),
                [](DepthFeaturePair const& p) -> float { return p.depth; } );

  //cv::normalize(depths,depths_norm,0, 255, cv::NORM_MINMAX, -1);

  depths = (depths / (float)params.NUM_DISPARITIES*params.COLOUR_SCALE) * 255.0f;
  std::transform(depths.begin<float>(), depths.end<float>(), depths_norm.begin<float>(),
             [](float f) -> float { return std::max(0.0f, std::min(f, 255.0f)); }); //clamp between 0 and 255


  float radius = 2;

  for(auto t : triangles){

    cv::Point2f vert1 = sparse_supports[t[0]].kp1.pt;
    cv::Point2f vert2 = sparse_supports[t[1]].kp1.pt;
    cv::Point2f vert3 = sparse_supports[t[2]].kp1.pt;

    int depth1_norm = static_cast<int>(depths_norm.at<float>(0,t[0]));
    int depth2_norm = static_cast<int>(depths_norm.at<float>(0,t[1]));
    int depth3_norm = static_cast<int>(depths_norm.at<float>(0,t[2]));

    //depth1_norm = std::min(depth1_norm, 255);
    //depth2_norm = std::min(depth2_norm, 255);
    //depth3_norm = std::min(depth3_norm, 255);

    //cv::Vec3b c1, c2, c3;
    //COLOUR_MAP.lookup(depth1_norm, c1);
    //COLOUR_MAP.lookup(depth2_norm, c2);
    //COLOUR_MAP.lookup(depth3_norm, c3);

    //dont draw mesh triangle if depth values are too different
    float filter_val = 1.8;
    if((float)(depth1_norm+1)/(float)(depth2_norm+1) > filter_val || (float)(depth2_norm+1)/(float)(depth1_norm+1) > filter_val){
      continue;
    }
    if((float)(depth1_norm+1)/(float)(depth3_norm+1) > filter_val || (float)(depth3_norm+1)/(float)(depth1_norm+1) > filter_val){
      continue;
    }
    if((float)(depth2_norm+1)/(float)(depth3_norm+1) > filter_val || (float)(depth3_norm+1)/(float)(depth2_norm+1) > filter_val){
      continue;
    }


    cv::Vec3b c1,c2,c3;
    COLOUR_MAP.lookup(depth1_norm, c1);
    COLOUR_MAP.lookup(depth2_norm, c2);
    COLOUR_MAP.lookup(depth3_norm, c3);
    //circle(img, vert1,radius, c1,cv::FILLED);
    //circle(img, vert2,radius, c2,cv::FILLED);
    //circle(img, vert3,radius, c3,cv::FILLED);


    drawLineColourMap(img,vert1,vert2, depth1_norm, depth2_norm);
    drawLineColourMap(img,vert2,vert3, depth2_norm, depth3_norm);
    drawLineColourMap(img,vert3,vert1, depth3_norm, depth1_norm);



  }

  //float radius = 2;
  //draw keypoints
  //for (int i = 0; i < sparse_supports.size();  i++){
  //  cv::Point2f vert = sparse_supports[i].kp1.pt;
  //  int depth_norm = static_cast<int>(depths_norm.at<float>(0,i));
  //  cv::Vec3b c1;
  //  COLOUR_MAP.lookup(depth_norm, c1);
  //  circle(img, vert,radius, c1, cv::FILLED);
  //}

  //imshow("Delaunay", img );
  //cv::waitKey(0);

}




/*

static cv::Mat linspace(float x0, float x1, int n)
{
    cv::Mat pts(n, 1, CV_32FC1);
    float step = (x1-x0)/(n-1);
    for(int i = 0; i < n; i++)
        pts.at<float>(i,0) = x0+i*step;
    return pts;
}

static cv::Mat argsort(cv::InputArray _src, bool ascending=true)
{
    cv::Mat src = _src.getMat();
    if (src.rows != 1 && src.cols != 1)
        CV_Error(cv::Error::StsBadArg, "cv::argsort only sorts 1D matrices.");
    int flags = cv::SORT_EVERY_ROW | (ascending ? cv::SORT_ASCENDING : cv::SORT_DESCENDING);
    cv::Mat sorted_indices;
    sortIdx(src.reshape(1,1),sorted_indices,flags);
    return sorted_indices;
}


static void sortMatrixRowsByIndices(cv::InputArray _src, cv::InputArray _indices, cv::OutputArray _dst)
{
    if(_indices.getMat().type() != CV_32SC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "cv::sortRowsByIndices only works on integer indices!");
    cv::Mat src = _src.getMat();
    std::vector<int> indices = _indices.getMat();
    _dst.create(src.rows, src.cols, src.type());
    cv::Mat dst = _dst.getMat();
    for(size_t idx = 0; idx < indices.size(); idx++) {
        cv::Mat originalRow = src.row(indices[idx]);
        cv::Mat sortedRow = dst.row((int)idx);
        originalRow.copyTo(sortedRow);
    }
}

static cv::Mat sortMatrixRowsByIndices(cv::InputArray src, cv::InputArray indices)
{
    cv::Mat dst;
    sortMatrixRowsByIndices(src, indices, dst);
    return dst;
}


template <typename _Tp> static
cv::Mat interp1_(const cv::Mat& X_, const cv::Mat& Y_, const cv::Mat& XI)
{
    int n = XI.rows;
    // sort input table
    std::vector<int> sort_indices = argsort(X_);

    cv::Mat X = sortMatrixRowsByIndices(X_,sort_indices);
    cv::Mat Y = sortMatrixRowsByIndices(Y_,sort_indices);
    // interpolated values
    cv::Mat yi = cv::Mat::zeros(XI.size(), XI.type());
    for(int i = 0; i < n; i++) {
        int low = 0;
        int high = X.rows - 1;
        // set bounds
        if(XI.at<_Tp>(i,0) < X.at<_Tp>(low, 0))
            high = 1;
        if(XI.at<_Tp>(i,0) > X.at<_Tp>(high, 0))
            low = high - 1;
        // binary search
        while((high-low)>1) {
            const int c = low + ((high - low) >> 1);
            if(XI.at<_Tp>(i,0) > X.at<_Tp>(c,0)) {
                low = c;
            } else {
                high = c;
            }
        }
        // linear interpolation
        yi.at<_Tp>(i,0) += Y.at<_Tp>(low,0)
        + (XI.at<_Tp>(i,0) - X.at<_Tp>(low,0))
        * (Y.at<_Tp>(high,0) - Y.at<_Tp>(low,0))
        / (X.at<_Tp>(high,0) - X.at<_Tp>(low,0));
    }
    return yi;
}

static cv::Mat interp1(cv::InputArray _x, cv::InputArray _Y, cv::InputArray _xi)
{
    // get matrices
    cv::Mat x = _x.getMat();
    cv::Mat Y = _Y.getMat();
    cv::Mat xi = _xi.getMat();
    // check types & alignment
    CV_Assert((x.type() == Y.type()) && (Y.type() == xi.type()));
    CV_Assert((x.cols == 1) && (x.rows == Y.rows) && (x.cols == Y.cols));
    // call templated interp1
    switch(x.type()) {
        case CV_8SC1: return interp1_<char>(x,Y,xi); break;
        case CV_8UC1: return interp1_<unsigned char>(x,Y,xi); break;
        case CV_16SC1: return interp1_<short>(x,Y,xi); break;
        case CV_16UC1: return interp1_<unsigned short>(x,Y,xi); break;
        case CV_32SC1: return interp1_<int>(x,Y,xi); break;
        case CV_32FC1: return interp1_<float>(x,Y,xi); break;
        case CV_64FC1: return interp1_<double>(x,Y,xi); break;
    }
    CV_Error(cv::Error::StsUnsupportedFormat, "");
}


cv::Mat linear_colormap(cv::InputArray X,
            cv::InputArray r, cv::InputArray g, cv::InputArray b,
            cv::InputArray xi) {
        cv::Mat lut, lut8;
        cv::Mat planes[] = {
                interp1(X, b, xi),
                interp1(X, g, xi),
                interp1(X, r, xi)};
        merge(planes, 3, lut);
        lut.convertTo(lut8, CV_8U, 255.);
        return lut8;
}

// Interpolates from a base colormap.
cv::Mat linear_colormap(cv::InputArray X,
        cv::InputArray r, cv::InputArray g, cv::InputArray b,
        int n) {
    return linear_colormap(X,r,g,b,linspace(0,1,n));
}

// Interpolates from a base colormap.
cv::Mat linear_colormap(cv::InputArray X,
        cv::InputArray r, cv::InputArray g, cv::InputArray b,
        float begin, float end, float n) {
    return linear_colormap(X,r,g,b,linspace(begin,end, cvRound(n)));
}



void generateColorMapLUT(cv::Mat & lut ){
  int n=256;
  cv::Mat X = linspace(0,1,256);
  // define the basemap
  float r[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00588235294117645f,0.02156862745098032f,0.03725490196078418f,0.05294117647058827f,0.06862745098039214f,0.084313725490196f,0.1000000000000001f,0.115686274509804f,0.1313725490196078f,0.1470588235294117f,0.1627450980392156f,0.1784313725490196f,0.1941176470588235f,0.2098039215686274f,0.2254901960784315f,0.2411764705882353f,0.2568627450980392f,0.2725490196078431f,0.2882352941176469f,0.303921568627451f,0.3196078431372549f,0.3352941176470587f,0.3509803921568628f,0.3666666666666667f,0.3823529411764706f,0.3980392156862744f,0.4137254901960783f,0.4294117647058824f,0.4450980392156862f,0.4607843137254901f,0.4764705882352942f,0.4921568627450981f,0.5078431372549019f,0.5235294117647058f,0.5392156862745097f,0.5549019607843135f,0.5705882352941174f,0.5862745098039217f,0.6019607843137256f,0.6176470588235294f,0.6333333333333333f,0.6490196078431372f,0.664705882352941f,0.6803921568627449f,0.6960784313725492f,0.7117647058823531f,0.7274509803921569f,0.7431372549019608f,0.7588235294117647f,0.7745098039215685f,0.7901960784313724f,0.8058823529411763f,0.8215686274509801f,0.8372549019607844f,0.8529411764705883f,0.8686274509803922f,0.884313725490196f,0.8999999999999999f,0.9156862745098038f,0.9313725490196076f,0.947058823529412f,0.9627450980392158f,0.9784313725490197f,0.9941176470588236f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9862745098039216f,0.9705882352941178f,0.9549019607843139f,0.93921568627451f,0.9235294117647062f,0.9078431372549018f,0.892156862745098f,0.8764705882352941f,0.8607843137254902f,0.8450980392156864f,0.8294117647058825f,0.8137254901960786f,0.7980392156862743f,0.7823529411764705f,0.7666666666666666f,0.7509803921568627f,0.7352941176470589f,0.719607843137255f,0.7039215686274511f,0.6882352941176473f,0.6725490196078434f,0.6568627450980391f,0.6411764705882352f,0.6254901960784314f,0.6098039215686275f,0.5941176470588236f,0.5784313725490198f,0.5627450980392159f,0.5470588235294116f,0.5313725490196077f,0.5156862745098039f,0.5f};
  float g[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001960784313725483f,0.01764705882352935f,0.03333333333333333f,0.0490196078431373f,0.06470588235294117f,0.08039215686274503f,0.09607843137254901f,0.111764705882353f,0.1274509803921569f,0.1431372549019607f,0.1588235294117647f,0.1745098039215687f,0.1901960784313725f,0.2058823529411764f,0.2215686274509804f,0.2372549019607844f,0.2529411764705882f,0.2686274509803921f,0.2843137254901961f,0.3f,0.3156862745098039f,0.3313725490196078f,0.3470588235294118f,0.3627450980392157f,0.3784313725490196f,0.3941176470588235f,0.4098039215686274f,0.4254901960784314f,0.4411764705882353f,0.4568627450980391f,0.4725490196078431f,0.4882352941176471f,0.503921568627451f,0.5196078431372548f,0.5352941176470587f,0.5509803921568628f,0.5666666666666667f,0.5823529411764705f,0.5980392156862746f,0.6137254901960785f,0.6294117647058823f,0.6450980392156862f,0.6607843137254901f,0.6764705882352942f,0.692156862745098f,0.7078431372549019f,0.723529411764706f,0.7392156862745098f,0.7549019607843137f,0.7705882352941176f,0.7862745098039214f,0.8019607843137255f,0.8176470588235294f,0.8333333333333333f,0.8490196078431373f,0.8647058823529412f,0.8803921568627451f,0.8960784313725489f,0.9117647058823528f,0.9274509803921569f,0.9431372549019608f,0.9588235294117646f,0.9745098039215687f,0.9901960784313726f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9901960784313726f,0.9745098039215687f,0.9588235294117649f,0.943137254901961f,0.9274509803921571f,0.9117647058823528f,0.8960784313725489f,0.8803921568627451f,0.8647058823529412f,0.8490196078431373f,0.8333333333333335f,0.8176470588235296f,0.8019607843137253f,0.7862745098039214f,0.7705882352941176f,0.7549019607843137f,0.7392156862745098f,0.723529411764706f,0.7078431372549021f,0.6921568627450982f,0.6764705882352944f,0.6607843137254901f,0.6450980392156862f,0.6294117647058823f,0.6137254901960785f,0.5980392156862746f,0.5823529411764707f,0.5666666666666669f,0.5509803921568626f,0.5352941176470587f,0.5196078431372548f,0.503921568627451f,0.4882352941176471f,0.4725490196078432f,0.4568627450980394f,0.4411764705882355f,0.4254901960784316f,0.4098039215686273f,0.3941176470588235f,0.3784313725490196f,0.3627450980392157f,0.3470588235294119f,0.331372549019608f,0.3156862745098041f,0.2999999999999998f,0.284313725490196f,0.2686274509803921f,0.2529411764705882f,0.2372549019607844f,0.2215686274509805f,0.2058823529411766f,0.1901960784313728f,0.1745098039215689f,0.1588235294117646f,0.1431372549019607f,0.1274509803921569f,0.111764705882353f,0.09607843137254912f,0.08039215686274526f,0.06470588235294139f,0.04901960784313708f,0.03333333333333321f,0.01764705882352935f,0.001960784313725483f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float b[] = {0.5f,0.5156862745098039f,0.5313725490196078f,0.5470588235294118f,0.5627450980392157f,0.5784313725490196f,0.5941176470588235f,0.6098039215686275f,0.6254901960784314f,0.6411764705882352f,0.6568627450980392f,0.6725490196078432f,0.6882352941176471f,0.7039215686274509f,0.7196078431372549f,0.7352941176470589f,0.7509803921568627f,0.7666666666666666f,0.7823529411764706f,0.7980392156862746f,0.8137254901960784f,0.8294117647058823f,0.8450980392156863f,0.8607843137254902f,0.8764705882352941f,0.892156862745098f,0.907843137254902f,0.9235294117647059f,0.9392156862745098f,0.9549019607843137f,0.9705882352941176f,0.9862745098039216f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9941176470588236f,0.9784313725490197f,0.9627450980392158f,0.9470588235294117f,0.9313725490196079f,0.915686274509804f,0.8999999999999999f,0.884313725490196f,0.8686274509803922f,0.8529411764705883f,0.8372549019607844f,0.8215686274509804f,0.8058823529411765f,0.7901960784313726f,0.7745098039215685f,0.7588235294117647f,0.7431372549019608f,0.7274509803921569f,0.7117647058823531f,0.696078431372549f,0.6803921568627451f,0.6647058823529413f,0.6490196078431372f,0.6333333333333333f,0.6176470588235294f,0.6019607843137256f,0.5862745098039217f,0.5705882352941176f,0.5549019607843138f,0.5392156862745099f,0.5235294117647058f,0.5078431372549019f,0.4921568627450981f,0.4764705882352942f,0.4607843137254903f,0.4450980392156865f,0.4294117647058826f,0.4137254901960783f,0.3980392156862744f,0.3823529411764706f,0.3666666666666667f,0.3509803921568628f,0.335294117647059f,0.3196078431372551f,0.3039215686274508f,0.2882352941176469f,0.2725490196078431f,0.2568627450980392f,0.2411764705882353f,0.2254901960784315f,0.2098039215686276f,0.1941176470588237f,0.1784313725490199f,0.1627450980392156f,0.1470588235294117f,0.1313725490196078f,0.115686274509804f,0.1000000000000001f,0.08431372549019622f,0.06862745098039236f,0.05294117647058805f,0.03725490196078418f,0.02156862745098032f,0.00588235294117645f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  // now build lookup table
  lut = linear_colormap(X,
          cv::Mat(256,1, CV_32FC1, (void*)r).clone(), // red
          cv::Mat(256,1, CV_32FC1, (void*)g).clone(), // green
          cv::Mat(256,1, CV_32FC1, (void*)b).clone(), // blue
          n);

}

*/


/*


    std::vector<int> R(3);
    std::vector<int> G(3);
    std::vector<int> B(3);

    //draw each edge in triangle
    for(int i = 0; i < 3; i++){
      float depth = sparse_supports[t[i]].depth;
      float fraction = depth/percent95;

      int c1R, c1G, c1B, c2R, c2G, c2B;

      float thresh = 0.5;

      if(fraction < thresh){
          c1R = 255; c1G = 0; c1B = 0;
          c2R = 180; c2G = 255; c2B = 0;

          R[i] =  (c2R-c1R) * fraction*(1/thresh) + c1R;
          G[i] =  (c2G-c1G) * fraction*(1/thresh) + c1G;
          B[i] =  (c2B-c1B) * fraction*(1/thresh) + c1B;
      }else{
          c1R = 180; c1G = 255; c1B = 0; //29, 221, 26
          c2R = 81; c2G = 103; c2B = 206; //37; 65; 206;

          R[i] =  (c2R-c1R) * fraction*thresh + c1R;
          G[i] =  (c2G-c1G) * fraction*thresh + c1G;
          B[i] =  (c2B-c1B) * fraction*thresh + c1B;
      }

      //correct features at the extremes that will be colored incorrectly based on above calcs.
      if(depth<percent5){
        R[i]=255;
        G[i]=0;
        B[i]=0;
      }
      if(depth>percent95){
        R[i]=81;
        G[i]=103;
        B[i]=206;
      }

    }

    cv::Scalar c1(B[0],G[0],R[0]);
    cv::Scalar c2(B[1],G[1],R[1]);
    cv::Scalar c3(B[2],G[2],R[2]);

    drawLineColourMap(img,vert1,vert2, c1,c2);
    drawLineColourMap(img,vert2,vert3, c2,c3);
    drawLineColourMap(img,vert3,vert1, c3,c1);

    */
