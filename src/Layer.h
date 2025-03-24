#pragma once

#include "KernelUtils.h"
#include <CL/opencl.hpp>
#include <vector>

class Layer {
public:
  std::string source = "";
  std::string backwardsWeightSource = "";
  std::string backwardsInputSource = "";
  std::string name = "";
  std::string backwardsWeightName = "";
  std::string backwardsInputName = "";
  int in_features = 0;
  int out_features = 0;
  int batch_size = 1;
  cl::Kernel kernel;
  cl::Kernel backwardsWeightKernel;
  cl::Kernel backwardsInputKernel;
  cl::Buffer Y_buf;
  cl::Buffer X_buf;
  cl::Buffer dY_buf;
  cl::Buffer dX_buf;
  cl::Program program;
  size_t launchConfig[2] = {0, 0};
  size_t backwardsInputLaunchConfig[2] = {0, 0};
  size_t backwardsWeightLaunchConfig[2] = {0, 0};

  virtual ~Layer() = default;
  virtual void setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                            const cl::Buffer &gt_buf = cl::Buffer()) = 0;
  virtual void setBackwardsWeightKernelArg(cl::Buffer &dLoss_buf,
                                           cl::Buffer &dX_buf,
                                           cl::Context ctx) = 0;
  virtual void setBackwardsInputKernelArg(cl::Buffer &dLoss_buf,
                                          cl::Buffer &dX_buf,
                                          cl::Context ctx) = 0;
  cl::Kernel getKernel(cl::Context ctx, cl::Platform platform,
                       cl::Device device, std::string &source,
                       std::string &kernelName, cl::Kernel &setKernel);
  cl::Kernel getForwardKernel(cl::Context ctx, cl::Platform platform,
                              cl::Device device);
  cl::Kernel getBackwardsWeightKernel(cl::Context ctx, cl::Platform platform,
                                      cl::Device device);
  cl::Kernel getBackwardsInputKernel(cl::Context ctx, cl::Platform platform,
                                     cl::Device device);
  cl::Program getProgram(cl::Context ctx, std::string &source);
};
