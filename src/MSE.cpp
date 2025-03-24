#include "MSE.h"

MSE::MSE(int in, int batch) {
  in_features = in;
  out_features = 1;
  batch_size = batch;
  source = readKernelFile("MSE.cl");
  backwardsWeightSource = readKernelFile("MSE_back_W.cl");
  backwardsInputSource = readKernelFile("MSE_back_X.cl");
  name = "MSE";
  backwardsWeightName = "MSE_back_W";
  backwardsInputName = "MSE_back_X";
  launchConfig[0] = batch;
  launchConfig[1] = 1;
}

MSE::~MSE() {}

void MSE::setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                       const cl::Buffer &gt) {
  kernel.setArg(0, X_buf);
  kernel.setArg(1, gt);
  kernel.setArg(2, Y_buf);
  kernel.setArg(3, batch_size);
  kernel.setArg(4, in_features);
}

void MSE::setBackwardsWeightKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                               cl::Context ctx){};

void MSE::setBackwardsInputKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                               cl::Context ctx){};


