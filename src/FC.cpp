#include "FC.h"

FC::FC(int in, int out, int batch, bool randomize)
    : in_features(in), out_features(out), batch_size(batch) {
  for (int i = 0; i < in * out; i++) {
    if (randomize)
      W.push_back((float)rand() / (float)RAND_MAX);
    else
      W.push_back(0);
  }
}

FC::~FC() {}

cl_program FC::getProgram(cl_context ctx) {
  const char *code = source.c_str();
  size_t size = source.size();
  return clCreateProgramWithSource(ctx, 1, &code, &size, nullptr);
}

cl_kernel FC::getKernel(cl_context ctx, cl_platform_id, cl_device_id device) {
  program = getProgram(ctx);
  cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  checkErr(err, "Failed to build program");

  kernel = clCreateKernel(program, "FC", &err);
  checkErr(err, "Failed to create kernel");

  Y_buf =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                     out_features * batch_size * sizeof(float), nullptr, &err);
  checkErr(err, "Failed to create Y_buf");
  return kernel;
}

void FC::getWeightBuffers(cl_context ctx) {
  if (!weightBuffer)
    weightBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  W.size() * sizeof(float), W.data(), nullptr);
}

void FC::setKernelArg(cl_mem &X_buf, cl_context ctx) {
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
