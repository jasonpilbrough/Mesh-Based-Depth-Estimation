#include "regularisers.h"
#include <iomanip>


void sandbox::Mesh_Regulariser::run_TV( std::vector<DepthFeaturePair> & sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> & sparse_supports_out){

  //auto start = std::chrono::system_clock::now();

  float σ = 0.125f; //0.025f
  float τ = 0.125f;
  float λ = 0.5f; //1.0
  float θ = 1.0f; //1
  int L = 500; //50


  //for(auto & dfp : sparse_supports_in){
  //  if(dfp.kp1.pt.x<700){
  //      dfp.depth = 32.0f;
  //  }else{
  //      dfp.depth = 0.0f;
  //  }

  //}
  //sparse_supports_in[210].depth = 64.0f;



  std::vector<float> z(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z.begin(),
                [](DepthFeaturePair const& p) -> float { return p.depth; } );


  std::vector<cv::KeyPoint> z_pts(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z_pts.begin(),
                [](DepthFeaturePair const& p) -> cv::KeyPoint { return p.kp1; } );


  std::vector<float> x(z);
  std::vector<float> p(edges.size(),0);
  std::vector<float> x_bar(x);



  for(int i = 0; i < L; i++){
    std::vector<float> x_prev(x);

    for(int i = 0; i < edges.size(); i++){
      float u_p = p[i] + σ * (x_bar[edges[i][1]] - x_bar[edges[i][0]]); // (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);
      p[i] = u_p/std::max(std::abs(u_p),1.0f);
    }

    for(int i = 0; i < edges.size(); i++){
      x[edges[i][0]] += τ * (p[i]); // (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);;
      x[edges[i][1]] -= τ * (p[i]); // (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);;
    }

    //L2 norm
    //std::transform(x.begin(), x.end(), z.begin(), x.begin(),
    //              [=](float const& x_i, float const& z_i) -> float { return (x_i+ λ * τ * z_i)/(1 + λ * τ); } );

    //L1 norm
    std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                  [=](float const& x_i, float const& z_i) -> float { return (x_i-z_i> λ*τ ? x_i - λ*τ : (x_i-z_i < -λ*τ ? x_i + λ*τ: z_i)) ;} );


    std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                  [=](float const& x_i, float const& x_prev_i) -> float { return  x_i+ θ*(x_i-x_prev_i); } );
  }



  //std::transform(sparse_supports_out.begin(), sparse_supports_out.end(), x.begin(), sparse_supports_out.begin(),
  //              [](DepthFeaturePair & support_pt_i, float const& x_i) -> DepthFeaturePair {support_pt_i.depth=x_i ;return support_pt_i;} );




  //std::cout << std::fixed;
  //std::cout << std::setprecision(2);
  for(int i = 0; i < sparse_supports_out.size(); i++){
    sparse_supports_out[i].depth = x[i];
    //std::cout << x[i] << " " << sparse_supports_in[i].depth << " "<<std::endl;
  }

  //auto end = std::chrono::system_clock::now();
  //std::chrono::duration<double> elapsed_seconds = end-start;
  //std::cout << elapsed_seconds.count() <<"s"<< std::endl;

}




void sandbox::Mesh_Regulariser::run_TGV( std::vector<DepthFeaturePair> & sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> & sparse_supports_out){

  //auto start = std::chrono::system_clock::now();

  float σ = 0.125f;
  float τ = 0.125f;
  float λ = 0.5f;
  float θ = 1.0f;
  float alpha1 = 0.3f;
  float alpha2 = 0.8f;
  int L = 500;


  //for(auto & dfp : sparse_supports_in){
  //  dfp.depth = 32.0f;
  //}
  //sparse_supports_in[50].depth = 64.0f;

  std::vector<float> z(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z.begin(),
                [](DepthFeaturePair const& p) -> float { return p.depth; } );

  std::vector<cv::KeyPoint> z_pts(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z_pts.begin(),
                [](DepthFeaturePair const& p) -> cv::KeyPoint { return p.kp1; } );

  std::vector<float> x(z);
  std::vector<float> p(edges.size(),0);
  std::vector<float> x_bar(x);

  std::vector<float> y(x.size(),0);
  std::vector<float> q(edges.size(),0);
  std::vector<float> y_bar(y);




  for(int i = 0; i < L; i++){
    std::vector<float> x_prev(x);
    std::vector<float> y_prev(y);
    std::vector<float> p_prev(p);
    std::vector<float> q_prev(q);

    for(int i = 0; i < edges.size(); i++){
      float u_p_1 = p[i] + σ * alpha1 * (x_bar[edges[i][1]] - x_bar[edges[i][0]] - y_bar[edges[i][0]]);

      //u_p_1 -= σ * alpha1 * (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);
      //u_p_1 -= σ * alpha1 * (z_pts[edges[i][1]].pt.y - z_pts[edges[i][0]].pt.y);

      p[i] = u_p_1/std::max(std::abs(u_p_1),1.0f);
      float u_p_2 = q[i] + σ * alpha2 * (y_bar[edges[i][1]] - y_bar[edges[i][0]]);
      q[i] = u_p_2/std::max(std::abs(u_p_2),1.0f);
    }




    for(int i = 0; i < edges.size(); i++){
      x[edges[i][0]] += τ * alpha1 * (p[i]);
      x[edges[i][1]] -= τ * alpha1 * (p[i]);
      y[edges[i][0]] += τ * alpha1 * (p_prev[i]);
      y[edges[i][1]] += τ * alpha1 * (p_prev[i]);
      y[edges[i][0]] += τ * alpha2 * (q_prev[i]);
      y[edges[i][1]] -= τ * alpha2 * (q_prev[i]);

      //y[edges[i][0]] += τ * alpha1 * (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);
      //y[edges[i][1]] -= τ * alpha1 * (z_pts[edges[i][1]].pt.y - z_pts[edges[i][0]].pt.y);

    }

    //L2 norm
    //std::transform(x.begin(), x.end(), z.begin(), x.begin(),
    //              [=](float const& x_i, float const& z_i) -> float { return (x_i+ λ * τ * z_i)/(1 + λ * τ); } );

    //L1 norm
    std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                  [=](float const& x_i, float const& z_i) -> float { return (x_i-z_i> λ*τ ? x_i - λ*τ : (x_i-z_i < -λ*τ ? x_i + λ*τ: z_i)) ;} );


    std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                  [=](float const& x_i, float const& x_prev_i) -> float { return x_i+ θ*(x_i-x_prev_i); } );

    std::transform(y.begin(), y.end(), y_prev.begin(), y_bar.begin(),
                  [=](float const& y_i, float const& y_prev_i) -> float { return y_i+θ*(y_i-y_prev_i); } );
  }



  //std::transform(sparse_supports_out.begin(), sparse_supports_out.end(), x.begin(), sparse_supports_out.begin(),
  //              [](DepthFeaturePair & support_pt_i, float const& x_i) -> DepthFeaturePair {support_pt_i.depth=x_i ;return support_pt_i;} );




  //std::cout << std::fixed;
  //std::cout << std::setprecision(2);
  for(int i = 0; i < sparse_supports_out.size(); i++){
    sparse_supports_out[i].depth = x[i];
    //std::cout << x[i] << " " << sparse_supports_in[i].depth << " "<<x[i]/sparse_supports_in[i].depth<<std::endl;
  }

  //auto end = std::chrono::system_clock::now();
  //std::chrono::duration<double> elapsed_seconds = end-start;
  //std::cout <<1/elapsed_seconds.count()<<"fps - "<< elapsed_seconds.count() <<"s"<< std::endl;



}


void sandbox::Mesh_Regulariser::run_logTV( std::vector<DepthFeaturePair> & sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> & sparse_supports_out){

  //auto start = std::chrono::system_clock::now();

  float σ = 0.125f;
  float τ = 0.125f;
  float λ = 1.0f;
  float θ = 1.0f;
  float beta = 1;
  int L = 25;
  int F = 10;

  float scaleFactor = 32;


  std::vector<float> z(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z.begin(),
                [&scaleFactor](DepthFeaturePair const& p) -> float { return p.depth / scaleFactor; } );


  std::vector<cv::KeyPoint> z_pts(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z_pts.begin(),
                [](DepthFeaturePair const& p) -> cv::KeyPoint { return p.kp1; } );


  std::vector<float> x(z);
  std::vector<float> p(edges.size(),0);
  std::vector<float> x_bar(x);

  std::vector<float> w(edges.size(),0);

  for(int j = 0; j < F; j++){

    for(int i = 0; i < edges.size(); i++){
	     w[i] = beta / (1 + beta * abs((x[edges[i][1]] - x[edges[i][0]])));
    }

    for(int i = 0; i < L; i++){
      std::vector<float> x_prev(x);

      for(int i = 0; i < edges.size(); i++){
        float u_p = p[i] + σ * (x_bar[edges[i][1]] - x_bar[edges[i][0]]) * w[i]; // (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);
        p[i] = u_p/std::max(std::abs(u_p),1.0f);
      }

      for(int i = 0; i < edges.size(); i++){
        x[edges[i][0]] += τ * (p[i]) * w[i]; // (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);;
        x[edges[i][1]] -= τ * (p[i]) * w[i]; // (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);;
      }

      //L2 norm
      //std::transform(x.begin(), x.end(), z.begin(), x.begin(),
      //              [=](float const& x_i, float const& z_i) -> float { return (x_i+ λ * τ * z_i)/(1 + λ * τ); } );

      //L1 norm
      std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                    [=](float const& x_i, float const& z_i) -> float { return (x_i-z_i> λ*τ ? x_i - λ*τ : (x_i-z_i < -λ*τ ? x_i + λ*τ: z_i)) ;} );


      std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                    [=](float const& x_i, float const& x_prev_i) -> float { return  x_i+ θ*(x_i-x_prev_i); } );

    }
}




  //std::cout << std::fixed;
  //std::cout << std::setprecision(2);
  for(int i = 0; i < sparse_supports_out.size(); i++){
    sparse_supports_out[i].depth = x[i] * scaleFactor;
    //std::cout << x[i] << " " << sparse_supports_in[i].depth << " "<<x[i]/sparse_supports_in[i].depth<<std::endl;
  }



}

void sandbox::Mesh_Regulariser::run_logTGV( std::vector<DepthFeaturePair> & sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> & sparse_supports_out){

  //TODO
}



/*

  //auto start = std::chrono::system_clock::now();

  float σ = 0.125f;
  float τ = 0.125f;
  float λ = 0.5f;
  float θ = 1.0f;
  float alpha1 = 0.3f;
  float alpha2 = 0.8f;
  float beta = 1;
  int L = 35;
  int F = 20;


  float scaleFactor = 32;


  std::vector<float> z(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z.begin(),
                [&scaleFactor](DepthFeaturePair const& p) -> float { return p.depth / scaleFactor; } );

  std::vector<cv::KeyPoint> z_pts(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z_pts.begin(),
                [](DepthFeaturePair const& p) -> cv::KeyPoint { return p.kp1; } );

  std::vector<float> x(z);
  std::vector<float> p(edges.size(),0);
  std::vector<float> x_bar(x);

  std::vector<float> y(x.size(),0);
  std::vector<float> q(edges.size(),0);
  std::vector<float> y_bar(y);


  std::vector<float> w(edges.size(),0);
  std::vector<float> wy(edges.size(),0);



for(int j = 0; j < F; j++){

  for(int i = 0; i < edges.size(); i++){
     w[i] = beta / (1 + beta * abs((x[edges[i][1]] - x[edges[i][0]] - y[edges[i][0]])));
     wy[i] = beta / (1 + beta * abs((y[edges[i][1]] - y[edges[i][0]])));
  }

  for(int k = 0; k < L; k++){
    std::vector<float> x_prev(x);
    std::vector<float> y_prev(y);
    std::vector<float> p_prev(p);
    std::vector<float> q_prev(q);

    for(int i = 0; i < edges.size(); i++){

      float u_p_1 = p[i] + σ * alpha1 * ((x_bar[edges[i][1]] - x_bar[edges[i][0]])*w[i] - y_bar[edges[i][0]]);
      p[i] = u_p_1/std::max(std::abs(u_p_1),1.0f);
      float u_p_2 = q[i] + σ * alpha2 * (y_bar[edges[i][1]] - y_bar[edges[i][0]]) * wy[i];
      q[i] = u_p_2/std::max(std::abs(u_p_2),1.0f);

    }




    for(int i = 0; i < edges.size(); i++){
      x[edges[i][0]] += τ * alpha1 * (p[i]) * w[i];
      x[edges[i][1]] -= τ * alpha1 * (p[i]) * w[i];
      y[edges[i][0]] += τ * alpha1 * (p_prev[i]);
      y[edges[i][1]] += τ * alpha1 * (p_prev[i]);
      y[edges[i][0]] += τ * alpha2 * (q_prev[i]) * wy[i];
      y[edges[i][1]] -= τ * alpha2 * (q_prev[i]) * wy[i];


    }

    //L2 norm
    //std::transform(x.begin(), x.end(), z.begin(), x.begin(),
    //              [=](float const& x_i, float const& z_i) -> float { return (x_i+ λ * τ * z_i)/(1 + λ * τ); } );

    //L1 norm
    std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                  [=](float const& x_i, float const& z_i) -> float { return (x_i-z_i> λ*τ ? x_i - λ*τ : (x_i-z_i < -λ*τ ? x_i + λ*τ: z_i)) ;} );


    std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                  [=](float const& x_i, float const& x_prev_i) -> float { return x_i+ θ*(x_i-x_prev_i); } );

    std::transform(y.begin(), y.end(), y_prev.begin(), y_bar.begin(),
                  [=](float const& y_i, float const& y_prev_i) -> float { return y_i+θ*(y_i-y_prev_i); } );
  }
}



  //std::transform(sparse_supports_out.begin(), sparse_supports_out.end(), x.begin(), sparse_supports_out.begin(),
  //              [](DepthFeaturePair & support_pt_i, float const& x_i) -> DepthFeaturePair {support_pt_i.depth=x_i ;return support_pt_i;} );




  //std::cout << std::fixed;
  //std::cout << std::setprecision(2);
  for(int i = 0; i < sparse_supports_out.size(); i++){
    sparse_supports_out[i].depth = x[i]*scaleFactor;
    //std::cout << x[i] << " " << sparse_supports_in[i].depth << " "<<x[i]/sparse_supports_in[i].depth<<std::endl;
  }

  //auto end = std::chrono::system_clock::now();
  //std::chrono::duration<double> elapsed_seconds = end-start;
  //std::cout <<1/elapsed_seconds.count()<<"fps - "<< elapsed_seconds.count() <<"s"<< std::endl;




*/


/*
void sandbox::NLTGV_Regulariser::run( std::vector<DepthFeaturePair> & sparse_supports_in, const std::vector<Edge> & edges, std::vector<DepthFeaturePair> & sparse_supports_out){

  //auto start = std::chrono::system_clock::now();

  float σ = 125;
  float τ = 0.001;
  float λ = 0.1f;
  float θ = 1.0f;
  float alpha1 = 0.3f;
  float alpha2 = 1.0f;
  int L = 1000;



  std::vector<float> z(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z.begin(),
                [](DepthFeaturePair const& p) -> float { return p.depth; } );

  std::vector<cv::KeyPoint> z_pts(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), z_pts.begin(),
                [](DepthFeaturePair const& p) -> cv::KeyPoint { return p.kp1; } );

  std::vector<float> x(z);
  std::vector<float> p(edges.size(),0);
  std::vector<float> x_bar(x);

  std::vector<float> y(x.size(),0);
  std::vector<float> q(edges.size(),0);
  std::vector<float> y_bar(y);

  std::vector<float> y2(x.size(),0);
  std::vector<float> q2(edges.size(),0);
  std::vector<float> y2_bar(y2);




  for(int i = 0; i < L; i++){
    std::vector<float> x_prev(x);
    std::vector<float> y_prev(y);
    std::vector<float> y2_prev(y2);
    std::vector<float> p_prev(p);
    std::vector<float> q_prev(q);
    std::vector<float> q2_prev(q2);

    for(int i = 0; i < edges.size(); i++){

      cv::Point2f vert1 = sparse_supports_in[edges[i][0]].kp1.pt;
      cv::Point2f vert2 = sparse_supports_in[edges[i][1]].kp1.pt;
      float x_diff = vert1.x - vert2.x;
      float y_diff = vert1.y - vert2.y;

      alpha1 = 1 / sqrt(x_diff*x_diff+y_diff*y_diff);

      //std::cout << alpha1 << std::endl;


      float u_p_1 = p[i] + σ * alpha1 * (x_bar[edges[i][1]] - x_bar[edges[i][0]] -  y_bar[edges[i][0]]*x_diff - y2_bar[edges[i][0]]*y_diff);
      p[i] = u_p_1/std::max(std::abs(u_p_1),1.0f);
      float u_p_2 = q[i] + σ * alpha2 * (y_bar[edges[i][1]] - y_bar[edges[i][0]]);
      q[i] = u_p_2/std::max(std::abs(u_p_2),1.0f);
      float  u_p_3 = q2[i] + σ * alpha2 * (y2_bar[edges[i][1]] - y2_bar[edges[i][0]]);
  		q2[i] = u_p_3/std::max(std::abs(u_p_3),1.0f);
    }





    for(int i = 0; i < edges.size(); i++){

      cv::Point2f vert1 = sparse_supports_in[edges[i][0]].kp1.pt;
      cv::Point2f vert2 = sparse_supports_in[edges[i][1]].kp1.pt;
      float x_diff = vert1.x - vert2.x;
      float y_diff = vert1.y - vert2.y;

      alpha1 = 1 / sqrt(x_diff*x_diff+y_diff*y_diff);

      x[edges[i][0]] += τ * alpha1 * (p[i]);
  		x[edges[i][1]] -= τ * alpha1 * (p[i]);

  		y[edges[i][0]] += τ * (alpha1 * p[i]) * x_diff;

  		y[edges[i][0]] += τ * (alpha2 * (q[i]));
  		y[edges[i][1]] -= τ * (alpha2 * (q[i]));

  		y2[edges[i][0]] += τ * (alpha1 * p[i]) * y_diff;

  		y2[edges[i][0]] += τ * alpha2 * (q2[i]);
  		y2[edges[i][1]] -= τ * alpha2 * (q2[i]);


    }

    //L2 norm
    //std::transform(x.begin(), x.end(), z.begin(), x.begin(),
    //              [=](float const& x_i, float const& z_i) -> float { return (x_i+ λ * τ * z_i)/(1 + λ * τ); } );

    //L1 norm
    std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                  [=](float const& x_i, float const& z_i) -> float { return (x_i-z_i> λ*τ ? x_i - λ*τ : (x_i-z_i < -λ*τ ? x_i + λ*τ: z_i)) ;} );



    std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                  [=](float const& x_i, float const& x_prev_i) -> float { return x_i+ θ*(x_i-x_prev_i); } );

    std::transform(y.begin(), y.end(), y_prev.begin(), y_bar.begin(),
                  [=](float const& y_i, float const& y_prev_i) -> float { return y_i+θ*(y_i-y_prev_i); } );

    std::transform(y2.begin(), y2.end(), y2_prev.begin(), y2_bar.begin(),
                  [=](float const& y2_i, float const& y2_prev_i) -> float { return y2_i+θ*(y2_i-y2_prev_i); } );
  }


  for(int i = 0; i < sparse_supports_out.size(); i++){
    sparse_supports_out[i].depth = x[i];
  }



}
*/


/*
  auto start = std::chrono::system_clock::now();
  int N = sparse_supports_in.size();

  //for(int i = 0; i < triangles.size(); i++){
  //  std::cout << triangles[i] <<std::endl;
  //}

  typedef Eigen::Triplet<float> T;
  std::vector<T> tripletList;
  tripletList.reserve(triangles.size()*6+N);

  for(int i = 0; i < N; i++){
      tripletList.push_back(T(i,i,-1));
  }

  for(int i = 0; i < triangles.size(); i++){
      int vtx_1_inx = triangles[i][0];
      int vtx_2_inx = triangles[i][1];
      int vtx_3_inx = triangles[i][2];

      cv::Point2f vert1 = sparse_supports_in[vtx_1_inx].kp1.pt;
      cv::Point2f vert2 = sparse_supports_in[vtx_2_inx].kp1.pt;
      cv::Point2f vert3 = sparse_supports_in[vtx_3_inx].kp1.pt;

      float dist12 = (vert1 - vert2).x*(vert1 - vert2).x + (vert1 - vert2).y*(vert1 - vert2).y;
      float dist13 = (vert1 - vert3).x*(vert1 - vert3).x + (vert1 - vert3).y*(vert1 - vert3).y;
      float dist23 = (vert2 - vert3).x*(vert2 - vert3).x + (vert2 - vert3).y*(vert2 - vert3).y;

      tripletList.push_back(T(vtx_1_inx,vtx_2_inx,1/dist12));
      tripletList.push_back(T(vtx_1_inx,vtx_3_inx,1/dist13));
      tripletList.push_back(T(vtx_2_inx,vtx_1_inx,1/dist12));
      tripletList.push_back(T(vtx_2_inx,vtx_3_inx,1/dist23));
      tripletList.push_back(T(vtx_3_inx,vtx_1_inx,1/dist13));
      tripletList.push_back(T(vtx_3_inx,vtx_2_inx,1/dist23));
  }


  Eigen::SparseMatrix<float> K(N,N);
  K.setFromTriplets(tripletList.begin(), tripletList.end());


  for (int i=0; i<K.outerSize(); ++i){
    float row_total = 0;
    for (Eigen::SparseMatrix<float>::InnerIterator it(K,i); it; ++it)
    {

      float val = it.value();
      if(val>0){ //only not diagonal entries
        row_total+=val;
      }

      //it.row();   // row index
      //it.col();   // col index (here it is equal to i)
      //it.index(); // inner index, here it is equal to it.row()

    }

    for (Eigen::SparseMatrix<float>::InnerIterator it(K,i); it; ++it)
    {
      float val = it.value();
      if(val<0){ //only diagonal entries
        it.valueRef() = it.value()*row_total;
      }
      //it.row();   // row index
      //it.col();   // col index (here it is equal to i)
      //it.index(); // inner index, here it is equal to it.row()

    }
  }

  std::cout << K << std::endl;


  Eigen::SparseMatrix<float> K_T = K.transpose();

  std::vector<float> depths(sparse_supports_in.size());
  std::transform( sparse_supports_in.begin(), sparse_supports_in.end(), depths.begin(),
                [](DepthFeaturePair const& p) -> float { return p.depth; } );


  float σ = 0.125f;
  float τ = 0.125f;
  float λ  =0.15f;
  float θ = 1.0f;
  int L = 200;

  // Initilise variables
  Eigen::VectorXf z = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(depths.data(), depths.size());
  Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(depths.data(), depths.size());
  Eigen::VectorXf p(sparse_supports_in.size());
  Eigen::VectorXf x_bar(x);
  Eigen::VectorXf x_prev(x);


  for(int i = 0; i < L; i++){
    x_prev = x;

    //dual step
    Eigen::VectorXf u_p = p  + σ * (K * x_bar);
    p = u_p.unaryExpr([](float const& t) -> float { return t / std::max(std::abs(t),1.0f); } );
    //std::vector<float> p_temp(p.size());
    //std::transform( &p, &p, p_temp.begin(), [](float const& t) -> float { return t / std::max(std::abs(t),1.0f); } );
    //p = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(p_temp.data(), p_temp.size());

    //primal step
    auto u_x = x - τ * (K_T * p);
    x = u_x + λ * τ * z/(1 + λ * τ);
    //std::cout << x << std::endl;

    //extra-gradient step
    x_bar = x + θ*(x-x_prev);
  }

  //std::cout << x << std::endl;


  for(int i = 0; i < sparse_supports_out.size(); i++){
    sparse_supports_out[i].depth = x[i];
    //std::cout << x[i] << " " << sparse_supports_in[i].depth << " "<<x[i]/sparse_supports_in[i].depth<<std::endl;
  }




  //u_x = x - τ *  (K.T * csr_matrix(p))
	//x = (u_x + λ * τ * z_mesh)/(1 + λ * τ)


  //for(int i = 0; i < p.size(); i++){
  //  p[i] = u_p[i] / std::max(std::abs(u_p[i]),1.0f);
  //}


  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << elapsed_seconds.count() <<"s"<< std::endl;

  */

/*

        flame::optimizers::nltgv2_l1_graph_regularizer::Graph graph_;
        int count = 0;
        for (DepthFeaturePair pair : support_points) {

          // Add new vertex to graph.
          flame::optimizers::nltgv2_l1_graph_regularizer::VertexHandle vtx_ii = boost::add_vertex(flame::optimizers::nltgv2_l1_graph_regularizer::VertexData(), graph_);

          // Initialize vertex data.
          auto& vdata = (graph_)[vtx_ii];
          vdata.id = count++;
          vdata.pos = pair.kp1.pt;
          vdata.data_term = pair.depth;
          vdata.data_weight = 1.0f;


          vdata.x = pair.depth;
          vdata.x_bar = vdata.x;
          vdata.x_prev = vdata.x;
        }


        using VtxIdxToHandle = std::unordered_map<uint32_t, flame::optimizers::nltgv2_l1_graph_regularizer::VertexHandle>;
        //using VtxHandleToIdx = std::unordered_map<flame::optimizers::nltgv2_l1_graph_regularizer::VertexHandle, uint32_t>;

        VtxIdxToHandle vtx_idx_to_handle;
        //VtxIdxToHandle handle_to_vtx_idx;

        flame::optimizers::nltgv2_l1_graph_regularizer::Graph::vertex_iterator vit, end;
        boost::tie(vit, end) = boost::vertices(graph_);
        count = 0;
        for ( ; vit != end; ++vit) {
          vtx_idx_to_handle[count] = *vit;
          //handle_to_vtx_idx[graph_[*vit].id] = count;
          count++;
        }

//
        // Add new edges.
        for (int ii = 0; ii < delaunay.edges().size(); ++ii) {
            flame::optimizers::nltgv2_l1_graph_regularizer::VertexHandle vtx_ii = vtx_idx_to_handle[delaunay.edges()[ii][0]];
            flame::optimizers::nltgv2_l1_graph_regularizer::VertexHandle vtx_jj = vtx_idx_to_handle[delaunay.edges()[ii][1]];

          // Compute edge length.
          cv::Point2f u_ii = support_points[delaunay.edges()[ii][0]].kp1.pt;
          cv::Point2f u_jj = support_points[delaunay.edges()[ii][1]].kp1.pt;
          cv::Point2f diff(u_ii - u_jj);
          float edge_length = sqrt(diff.x*diff.x + diff.y*diff.y);

          //std::cout << vtx_ii << std::endl;
          //std::cout << delaunay.edges()[ii][1] << std::endl;

          if (!boost::edge(vtx_ii, vtx_jj, graph_).second) {
            // Add edge to graph if new.
            boost::add_edge(vtx_ii, vtx_jj, flame::optimizers::nltgv2_l1_graph_regularizer::EdgeData(), graph_);
          }

          // Initialize edge data.
          const auto& epair = boost::edge(vtx_ii, vtx_jj, graph_);
          auto& edata = (graph_)[epair.first];
          edata.alpha = 1.0f / edge_length;
          edata.beta = 1.0f;
          edata.valid = true;
        }
//


        std::cout <<"Intital data cost: "<< dataCost(flame::Params().rparams,graph_) << "  Inital smoothness cost: "<<smoothnessCost(flame::Params().rparams,graph_)<<std::endl;
        //std::cout <<"Smoothness cost before:"<<  << std::endl;

        for(int i = 0; i < 200; i++){
          flame::optimizers::nltgv2_l1_graph_regularizer::step(flame::Params().rparams, &graph_);
          //std::cout <<"Data cost: "<< dataCost(flame::Params().rparams,graph_) << "  Smoothness cost: "<<smoothnessCost(flame::Params().rparams,graph_)<<std::endl;

        }

        //flame::optimizers::nltgv2_l1_graph_regularizer::Graph::vertex_iterator vit, end;
        boost::tie(vit, end) = boost::vertices(graph_);

        count =0;
        for ( ; vit != end; ++vit) {
          auto& vtx = (graph_)[*vit];
          //std::cout <<vtx.x << std::endl;
          //std::cout <<support_points[vtx.id].depth <<" "<< vtx.x<< std::endl;
          //std::cout <<support_points[vtx.id].depth <<" "<< vtx.x<< std::endl;
          support_points[vtx.id].depth = vtx.x;
        }

        FLAME_ASSERT(delaunay.edges().size() == boost::num_edges(graph_));

        delaunay.drawWireframe(support_points,triangles,support_points_img);

        cv::Mat img_dense2 = cv::Mat(img_l.size().height, img_l.size().width, CV_32FC1, cv::Scalar(0));
        interpolateMesh(triangles,support_points,img_dense2);
        cv::normalize(img_dense2, img_dense2, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat img_color2;
        COLOUR_MAP.applyColourMap(img_dense2, img_color2);
        imshow("Dense2", img_color2);

*/
