#include "Sigmoid.h"

Sigmoid::Sigmoid(int in, int batch) {
  in_features = in;
  out_features = in;
  batch_size = batch;
  source = readKernelFile("Sigmoid.cl");
  name = "Sigmoid";
}

Sigmoid::~Sigmoid() {}

void Sigmoid::setKernelArg(cl_mem &X_buf, cl_context ctx) {
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &Y_buf);
  clSetKernelArg(kernel, 2, sizeof(int), &batch_size);
  clSetKernelArg(kernel, 3, sizeof(int), &in_features);
  clSetKernelArg(kernel, 4, sizeof(int), &out_features);
}
