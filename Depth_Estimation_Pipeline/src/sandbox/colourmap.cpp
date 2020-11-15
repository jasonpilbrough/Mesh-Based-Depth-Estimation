#include "colourmap.h"


/* ADAPTED FROM OPENCV */

static cv::Mat sandbox::linspace(float x0, float x1, int n)
{
    cv::Mat pts(n, 1, CV_32FC1);
    float step = (x1-x0)/(n-1);
    for(int i = 0; i < n; i++)
        pts.at<float>(i,0) = x0+i*step;
    return pts;
}

static cv::Mat sandbox::argsort(cv::InputArray & _src, bool ascending)
{
    cv::Mat src = _src.getMat();
    if (src.rows != 1 && src.cols != 1)
        CV_Error(cv::Error::StsBadArg, "cv::argsort only sorts 1D matrices.");
    int flags = cv::SORT_EVERY_ROW | (ascending ? cv::SORT_ASCENDING : cv::SORT_DESCENDING);
    cv::Mat sorted_indices;
    sortIdx(src.reshape(1,1),sorted_indices,flags);
    return sorted_indices;
}


static void sandbox::sortMatrixRowsByIndices(cv::InputArray & _src, cv::InputArray & _indices, cv::OutputArray & _dst)
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

static cv::Mat sandbox::sortMatrixRowsByIndices(cv::InputArray & src, cv::InputArray & indices)
{
    cv::Mat dst;
    sortMatrixRowsByIndices(src, indices, dst);
    return dst;
}


template <typename _Tp> static
cv::Mat sandbox::interp1_(const cv::Mat& X_, const cv::Mat& Y_, const cv::Mat& XI)
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

static cv::Mat sandbox::interp1(cv::InputArray & _x, cv::InputArray & _Y, cv::InputArray & _xi)
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





sandbox::ColourMap::ColourMap() : ColourMap(256) {}

sandbox::ColourMap::ColourMap(const int n){

  static const float r[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00588235294117645f,0.02156862745098032f,0.03725490196078418f,0.05294117647058827f,0.06862745098039214f,0.084313725490196f,0.1000000000000001f,0.115686274509804f,0.1313725490196078f,0.1470588235294117f,0.1627450980392156f,0.1784313725490196f,0.1941176470588235f,0.2098039215686274f,0.2254901960784315f,0.2411764705882353f,0.2568627450980392f,0.2725490196078431f,0.2882352941176469f,0.303921568627451f,0.3196078431372549f,0.3352941176470587f,0.3509803921568628f,0.3666666666666667f,0.3823529411764706f,0.3980392156862744f,0.4137254901960783f,0.4294117647058824f,0.4450980392156862f,0.4607843137254901f,0.4764705882352942f,0.4921568627450981f,0.5078431372549019f,0.5235294117647058f,0.5392156862745097f,0.5549019607843135f,0.5705882352941174f,0.5862745098039217f,0.6019607843137256f,0.6176470588235294f,0.6333333333333333f,0.6490196078431372f,0.664705882352941f,0.6803921568627449f,0.6960784313725492f,0.7117647058823531f,0.7274509803921569f,0.7431372549019608f,0.7588235294117647f,0.7745098039215685f,0.7901960784313724f,0.8058823529411763f,0.8215686274509801f,0.8372549019607844f,0.8529411764705883f,0.8686274509803922f,0.884313725490196f,0.8999999999999999f,0.9156862745098038f,0.9313725490196076f,0.947058823529412f,0.9627450980392158f,0.9784313725490197f,0.9941176470588236f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9862745098039216f,0.9705882352941178f,0.9549019607843139f,0.93921568627451f,0.9235294117647062f,0.9078431372549018f,0.892156862745098f,0.8764705882352941f,0.8607843137254902f,0.8450980392156864f,0.8294117647058825f,0.8137254901960786f,0.7980392156862743f,0.7823529411764705f,0.7666666666666666f,0.7509803921568627f,0.7352941176470589f,0.719607843137255f,0.7039215686274511f,0.6882352941176473f,0.6725490196078434f,0.6568627450980391f,0.6411764705882352f,0.6254901960784314f,0.6098039215686275f,0.5941176470588236f,0.5784313725490198f,0.5627450980392159f,0.5470588235294116f,0.5313725490196077f,0.5156862745098039f,0.5f};
  static const float g[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001960784313725483f,0.01764705882352935f,0.03333333333333333f,0.0490196078431373f,0.06470588235294117f,0.08039215686274503f,0.09607843137254901f,0.111764705882353f,0.1274509803921569f,0.1431372549019607f,0.1588235294117647f,0.1745098039215687f,0.1901960784313725f,0.2058823529411764f,0.2215686274509804f,0.2372549019607844f,0.2529411764705882f,0.2686274509803921f,0.2843137254901961f,0.3f,0.3156862745098039f,0.3313725490196078f,0.3470588235294118f,0.3627450980392157f,0.3784313725490196f,0.3941176470588235f,0.4098039215686274f,0.4254901960784314f,0.4411764705882353f,0.4568627450980391f,0.4725490196078431f,0.4882352941176471f,0.503921568627451f,0.5196078431372548f,0.5352941176470587f,0.5509803921568628f,0.5666666666666667f,0.5823529411764705f,0.5980392156862746f,0.6137254901960785f,0.6294117647058823f,0.6450980392156862f,0.6607843137254901f,0.6764705882352942f,0.692156862745098f,0.7078431372549019f,0.723529411764706f,0.7392156862745098f,0.7549019607843137f,0.7705882352941176f,0.7862745098039214f,0.8019607843137255f,0.8176470588235294f,0.8333333333333333f,0.8490196078431373f,0.8647058823529412f,0.8803921568627451f,0.8960784313725489f,0.9117647058823528f,0.9274509803921569f,0.9431372549019608f,0.9588235294117646f,0.9745098039215687f,0.9901960784313726f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9901960784313726f,0.9745098039215687f,0.9588235294117649f,0.943137254901961f,0.9274509803921571f,0.9117647058823528f,0.8960784313725489f,0.8803921568627451f,0.8647058823529412f,0.8490196078431373f,0.8333333333333335f,0.8176470588235296f,0.8019607843137253f,0.7862745098039214f,0.7705882352941176f,0.7549019607843137f,0.7392156862745098f,0.723529411764706f,0.7078431372549021f,0.6921568627450982f,0.6764705882352944f,0.6607843137254901f,0.6450980392156862f,0.6294117647058823f,0.6137254901960785f,0.5980392156862746f,0.5823529411764707f,0.5666666666666669f,0.5509803921568626f,0.5352941176470587f,0.5196078431372548f,0.503921568627451f,0.4882352941176471f,0.4725490196078432f,0.4568627450980394f,0.4411764705882355f,0.4254901960784316f,0.4098039215686273f,0.3941176470588235f,0.3784313725490196f,0.3627450980392157f,0.3470588235294119f,0.331372549019608f,0.3156862745098041f,0.2999999999999998f,0.284313725490196f,0.2686274509803921f,0.2529411764705882f,0.2372549019607844f,0.2215686274509805f,0.2058823529411766f,0.1901960784313728f,0.1745098039215689f,0.1588235294117646f,0.1431372549019607f,0.1274509803921569f,0.111764705882353f,0.09607843137254912f,0.08039215686274526f,0.06470588235294139f,0.04901960784313708f,0.03333333333333321f,0.01764705882352935f,0.001960784313725483f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const float b[] = {0.5f,0.5156862745098039f,0.5313725490196078f,0.5470588235294118f,0.5627450980392157f,0.5784313725490196f,0.5941176470588235f,0.6098039215686275f,0.6254901960784314f,0.6411764705882352f,0.6568627450980392f,0.6725490196078432f,0.6882352941176471f,0.7039215686274509f,0.7196078431372549f,0.7352941176470589f,0.7509803921568627f,0.7666666666666666f,0.7823529411764706f,0.7980392156862746f,0.8137254901960784f,0.8294117647058823f,0.8450980392156863f,0.8607843137254902f,0.8764705882352941f,0.892156862745098f,0.907843137254902f,0.9235294117647059f,0.9392156862745098f,0.9549019607843137f,0.9705882352941176f,0.9862745098039216f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9941176470588236f,0.9784313725490197f,0.9627450980392158f,0.9470588235294117f,0.9313725490196079f,0.915686274509804f,0.8999999999999999f,0.884313725490196f,0.8686274509803922f,0.8529411764705883f,0.8372549019607844f,0.8215686274509804f,0.8058823529411765f,0.7901960784313726f,0.7745098039215685f,0.7588235294117647f,0.7431372549019608f,0.7274509803921569f,0.7117647058823531f,0.696078431372549f,0.6803921568627451f,0.6647058823529413f,0.6490196078431372f,0.6333333333333333f,0.6176470588235294f,0.6019607843137256f,0.5862745098039217f,0.5705882352941176f,0.5549019607843138f,0.5392156862745099f,0.5235294117647058f,0.5078431372549019f,0.4921568627450981f,0.4764705882352942f,0.4607843137254903f,0.4450980392156865f,0.4294117647058826f,0.4137254901960783f,0.3980392156862744f,0.3823529411764706f,0.3666666666666667f,0.3509803921568628f,0.335294117647059f,0.3196078431372551f,0.3039215686274508f,0.2882352941176469f,0.2725490196078431f,0.2568627450980392f,0.2411764705882353f,0.2254901960784315f,0.2098039215686276f,0.1941176470588237f,0.1784313725490199f,0.1627450980392156f,0.1470588235294117f,0.1313725490196078f,0.115686274509804f,0.1000000000000001f,0.08431372549019622f,0.06862745098039236f,0.05294117647058805f,0.03725490196078418f,0.02156862745098032f,0.00588235294117645f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


    cv::Mat X =  sandbox::linspace(0,1,256);
    this->_lut = sandbox::ColourMap::linear_colormap(X,
            cv::Mat(256,1, CV_32FC1, (void*)r).clone(), // red
            cv::Mat(256,1, CV_32FC1, (void*)g).clone(), // green
            cv::Mat(256,1, CV_32FC1, (void*)b).clone(), // blue
            n);
}


cv::Mat sandbox::ColourMap::linear_colormap(cv::InputArray & X,
            cv::InputArray & r, cv::InputArray & g, cv::InputArray & b,
            cv::InputArray & xi) {

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
cv::Mat sandbox::ColourMap::linear_colormap(cv::InputArray & X,
        cv::InputArray & r, cv::InputArray & g, cv::InputArray & b,
        const int n) {
    return linear_colormap(X,r,g,b,linspace(0,1,n));
}

// Interpolates from a base colormap.
cv::Mat sandbox::ColourMap::linear_colormap(cv::InputArray & X,
        cv::InputArray & r, cv::InputArray & g, cv::InputArray & b,
        const float begin, const float end, const float n) {
    return linear_colormap(X,r,g,b,linspace(begin,end, cvRound(n)));
}

//use default cv colour map for dense images
void sandbox::ColourMap::applyColourMap(cv::InputArray & src, cv::OutputArray & dst) const{
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
}

//lookup_value must be normalised between 0 and 255
void sandbox::ColourMap::lookup(const int lookup_value, cv::Vec3b & colour_out) const{
  colour_out = _lut.at<cv::Vec3b>(lookup_value);
}


void sandbox::ColourMap::lookup2(const float lookup_value, cv::Vec3b & c) const {
  c = cv::Vec3b(255, 255, 255);  // white
  float dv;
  float vmin = 0.0f;
  float vmax = 255.0f;

  float v = lookup_value;


  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    c[2] = 0;
    c[1] = static_cast<uint8_t>(255 * (4 * (v - vmin) / dv));
  } else if (v < (vmin + 0.5 * dv)) {
    c[2] = 0;
    c[0] = static_cast<uint8_t>(255 * (1 + 4 * (vmin + 0.25 * dv - v) / dv));;
  } else if (v < (vmin + 0.75 * dv)) {
    c[2] = static_cast<uint8_t>(255 * (4 * (v - vmin - 0.5 * dv) / dv));
    c[0] = 0;
  } else {
    c[1] = static_cast<uint8_t>(255 * (1 + 4 * (vmin + 0.75 * dv - v) / dv));
    c[0] = 0;
  }

}


void sandbox::ColourMap::lookup_alt(int lookup_value, cv::Vec3b & colour_out) const{

  int c1R, c1G, c1B, c2R, c2G, c2B;
  float R, G, B;

  float thresh = 0.1;
  float fraction = lookup_value / 255.0;

  if(fraction < thresh){
      c1R = 255; c1G = 0; c1B = 0;
      c2R = 180; c2G = 255; c2B = 0;

      R =  (c2R-c1R) * fraction*(1/thresh) + c1R;
      G =  (c2G-c1G) * fraction*(1/thresh) + c1G;
      B =  (c2B-c1B) * fraction*(1/thresh) + c1B;

      std::cout << fraction << std::endl;
  }else{
      c1R = 180; c1G = 255; c1B = 0; //29, 221, 26
      c2R = 81; c2G = 103; c2B = 206; //37; 65; 206;

      R =  (c2R-c1R) * fraction*thresh + c1R;
      G =  (c2G-c1G) * fraction*thresh + c1G;
      B =  (c2B-c1B) * fraction*thresh + c1B;

  }


  colour_out[0] = static_cast<char>(B);
  colour_out[1] = static_cast<char>(G);
  colour_out[2] = static_cast<char>(R);



}
