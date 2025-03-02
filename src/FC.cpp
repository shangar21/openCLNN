#include "FC.h"

FC::FC(int in, int out, int batch, bool randomize) {
  in_features = in;
  out_features = out;
  batch_size = batch;
  source = readKernelFile("FC.cl");
  name = "FC";
  for (int i = 0; i < in * out; i++) {
    if (randomize)
      W.push_back((float)rand() / (float)RAND_MAX);
    else
      W.push_back(0);
  }
}

FC::~FC() {}

void FC::getWeightBuffers(cl_context ctx) {
  if (!weightBuffer)
    weightBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  W.size() * sizeof(float), W.data(), nullptr);
}

void FC::setKernelArg(cl_mem &X_buf, cl_context ctx) {
  getWeightBuffers(ctx);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightBuffer);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &Y_buf);
  clSetKernelArg(kernel, 3, sizeof(int), &batch_size);
  clSetKernelArg(kernel, 4, sizeof(int), &in_features);
  clSetKernelArg(kernel, 5, sizeof(int), &out_features);
}

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
