#pragma once

#include "KernelUtils.h"
#include "Layer.h"
#include <memory>
#include <ranges>

class Module {

public:
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Buffer Y_buf;
  std::vector<float> Y;
  std::vector<float> lossVals;

  std::vector<std::shared_ptr<Layer>> fullyConnectedLayers;
  std::shared_ptr<Layer> lossFunction;
  int inputSize = 0;

  Module();
  ~Module();

  void addLayer(std::shared_ptr<Layer> layer);
  void setLoss(std::shared_ptr<Layer> lossFunc);
  void forward(std::vector<float> X);
  void loss(std::vector<float> gts);
  void backwards();
  std::vector<float> getOutput();
};
