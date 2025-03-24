#include "FC.h"

FC::FC(int in, int out, int batch, bool randomize) {
  in_features = in;
  out_features = out;
  batch_size = batch;
  source = readKernelFile("FC.cl");
  backwardsSource = readKernelFile("FC_back.cl");
  name = "FC";
  backwardsName = "FC_back";
  launchConfig[0] = batch;
  launchConfig[1] = out;
	backwardsLaunchConfig[0] = batch;
  for (int i = 0; i < in * out; i++) {
    if (randomize)
      W.push_back((float)rand() / (float)RAND_MAX);
    else
      W.push_back(0);
  }
}

FC::~FC() {}

void FC::getWeightBuffers(cl::Context ctx) {
  if (!weightBuffer())
    weightBuffer = cl::Buffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              W.size() * sizeof(float), W.data());
}

void FC::setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                      const cl::Buffer &gt_buf) {
  getWeightBuffers(ctx);
  kernel.setArg(0, X_buf);
  kernel.setArg(1, weightBuffer);
  kernel.setArg(2, Y_buf);
  kernel.setArg(3, batch_size);
  kernel.setArg(4, in_features);
  kernel.setArg(5, out_features);
}

void FC::setBackwardsKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                               cl::Context ctx){};

std::ostream &operator<<(std::ostream &os, const FC &fc) {
  os << "Fully Connected Layer (" << fc.in_features << " X " << fc.out_features
     << ") "
     << "Batch Size: " << fc.batch_size << "\n";
  os << "Weights: \n";
  for (int i = 0; i < fc.in_features; i++) {
    for (int j = 0; j < fc.out_features; j++) {
      os << fc.W[i * fc.out_features + j] << " ";
    }
    os << "\n";
  }

  return os;
}
