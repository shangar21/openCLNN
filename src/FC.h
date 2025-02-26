#pragma once

#include "KernelUtils.h"
#include <CL/cl.h>
#include <vector>

class FC {
public:
  std::string source = readKernelFile("FC.cl");
  std::string name = "FC";
  const int in_features = 0;
  const int out_features = 0;
  const int batch_size = 1;
  cl_kernel kernel;
  std::vector<float> W;
  cl_mem Y_buf;
  std::vector<cl_mem> weightBuffers;

  FC(int in, int out, int batch_size = 1, bool randomize = true);
  ~FC();
  cl_kernel getKernel(cl_context ctx, cl_platform_id platform,
                      cl_device_id device);
  void getWeightBuffers(cl_context ctx);
  friend std::ostream &operator<<(std::ostream &os, const FC &fc);
  void setKernelArg(cl_mem &X_buf, cl_context ctx);

private:
  cl_program program = nullptr;
  cl_program getProgram(cl_context ctx);
};
