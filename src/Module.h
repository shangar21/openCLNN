#pragma once

#include "FC.h"
#include "KernelUtils.h"
#include <CL/cl.h>

class Module {

public:
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem Y_buf;
  std::vector<float> Y;

  std::vector<FC> fullyConnectedLayers;
  int inputSize = 0;

  Module();
  ~Module();

  void addLayer(FC &&fc);
  void forward(std::vector<float> X);
  std::vector<float> getOutput();
};
