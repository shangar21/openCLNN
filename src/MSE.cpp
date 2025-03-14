#include "MSE.h"

MSE::MSE(int in, int batch) {
  in_features = in;
  out_features = 1;
  batch_size = batch;
  source = readKernelFile("MSE.cl");
  backwardsSource = readKernelFile("MSE_back.cl");
  name = "MSE";
  backwardsName = "MSE_back";
  launchConfig[0] = batch;
  launchConfig[1] = in;
}

MSE::~MSE() {}

void MSE::setKernelArg(cl_mem &X_buf, cl_context ctx, const cl_mem &gt) {
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &gt);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &Y_buf);
  clSetKernelArg(kernel, 3, sizeof(int), &batch_size);
  clSetKernelArg(kernel, 4, sizeof(int), &in_features);
}

void MSE::setBackwardsKernelArg(cl_mem &dLoss_buf, cl_context ctx){};
