#include "FC.h"

FC::FC(int in, int out, int batch, bool randomize)
    : in_features(in), out_features(out), batch_size(batch) {
  for (int i = 0; i < in * out; i++) {
    if (randomize)
      W.push_back((float)rand() / RAND_MAX);
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

cl_kernel FC::getKernel(cl_context ctx, cl_platform_id platform,
                        cl_device_id device) {
  if (!program)
    program = getProgram(ctx);
  checkErr(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr),
           "Build FC Program");
  cl_kernel k = clCreateKernel(program, "FC", nullptr);
  kernel = k;

  std::vector<float> Y(out_features * batch_size, 0.0f);

  cl_int err;
  Y_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         Y.size() * sizeof(float), Y.data(), &err);
  checkErr(err, "Create Y_buf");

  if (!Y_buf) {
    throw std::runtime_error("Y_buf is NULL despite CL_SUCCESS");
  }

  return k;
}

void FC::getWeightBuffers(cl_context ctx) {
  weightBuffers.push_back(
      clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     W.size() * sizeof(float), W.data(), nullptr));
}

void FC::setKernelArg(cl_mem &X_buf, cl_context ctx) {

  int i = 0;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightBuffers[0]);
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
      os << fc.W[i * fc.in_features + j] << " ";
    }
    os << "\n";
  }

  return os;
}
