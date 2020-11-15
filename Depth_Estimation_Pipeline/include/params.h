#ifndef _SANDBOX_PARAMS
#define _SANDBOX_PARAMS


namespace sandbox{

  struct Params{
      //SPARSE MATCHING
      int MIN_HESSIAN = 1;
      int GRID_SIZE = 16; //in px
      int IMG_PADDING = 16; //in px

      //DENSE MATCHING
      int BLOCK_SIZE = 15;
      int NUM_DISPARITIES = 32;

      float COLOUR_SCALE = 1.5; //1.12

      //REGULARISATION
      float σ = 0.125f;
      float τ = 0.125f;
      float λ = 0.5f; //TGV:1.0, TV:0.5
      float θ = 1.0f;
      float alpha1 = 0.3f;
      float alpha2 = 0.8f;
      int L = 500;


  };

}

#endif
