#pragma once

#include "Layer.h"
#include "FC.h"
#include "KernelUtils.h"
#include <memory>

class Module {

public:
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem Y_buf;
  std::vector<float> Y;

  std::vector<std::shared_ptr<Layer>> fullyConnectedLayers;
  int inputSize = 0;

  Module();
  ~Module();

  void addLayer(std::shared_ptr<Layer> layer);
  void forward(std::vector<float> X);
  std::vector<float> getOutput();
};
