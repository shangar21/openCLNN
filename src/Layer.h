#pragma once

#include "KernelUtils.h"
#include <CL/opencl.hpp>
#include <vector>

class Layer {
public:
  std::string source = "";
  std::string backwardsSource = "";
  std::string name = "";
  std::string backwardsName = "";
  int in_features = 0;
  int out_features = 0;
  int batch_size = 1;
  cl::Kernel kernel;
  cl::Kernel backwardsKernel;
  cl::Buffer Y_buf;
  cl::Buffer X_buf;
  cl::Buffer dY_buf;
  cl::Buffer dX_buf;
  cl::Program program;
  size_t launchConfig[2] = {0, 0};

  virtual ~Layer() = default;
  virtual void setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                            const cl::Buffer &gt_buf = cl::Buffer()) = 0;
  virtual void setBackwardsKernelArg(cl::Buffer &dLoss_buf, cl::Context ctx) = 0;
  cl::Kernel getKernel(cl::Context ctx, cl::Platform platform,
                      cl::Device device, bool backwards = false);
  cl::Program getProgram(cl::Context ctx, bool backwards = false);
};
