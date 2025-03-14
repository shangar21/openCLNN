#pragma once

#include "KernelUtils.h"
#include <CL/cl.h>
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
  cl_kernel kernel = nullptr;
	cl_kernel backwardsKernel = nullptr;
  cl_mem Y_buf = nullptr;
  cl_mem dY_buf = nullptr;
  cl_mem dX_buf = nullptr;
  cl_program program = nullptr;
  size_t launchConfig[2] = {0, 0};

  virtual ~Layer() = default;
  virtual void setKernelArg(cl_mem &X_buf, cl_context ctx,
                            const cl_mem &gt_buf = nullptr) = 0;
	virtual void setBackwardsKernelArg(cl_mem &dLoss_buf, cl_context ctx) = 0;
  cl_kernel getKernel(cl_context ctx, cl_platform_id platform,
                      cl_device_id device, bool backwards = false);
  cl_program getProgram(cl_context ctx, bool backwards = false);
};
