#include "Sigmoid.h"

Sigmoid::Sigmoid(int in, int batch) {
  in_features = in;
  out_features = in;
  batch_size = batch;
  source = readKernelFile("Sigmoid.cl");
  backwardsSource = readKernelFile("Sigmoid_back.cl");
  name = "Sigmoid";
  backwardsSource = "Sigmoid_back";
  launchConfig[0] = batch;
  launchConfig[1] = in;
  backwardsLaunchConfig[0] = batch;
}

Sigmoid::~Sigmoid() {}

void Sigmoid::setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                           const cl::Buffer &gt_buf) {
  kernel.setArg(0, X_buf);
  kernel.setArg(1, Y_buf);
  kernel.setArg(2, batch_size);
  kernel.setArg(3, in_features);
  kernel.setArg(4, out_features);
}

void Sigmoid::setBackwardsKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                                    cl::Context ctx){};
