#include "FC.h"
#include "MSE.h"
#include "Module.h"
#include "Optimizer.h"
#include "Sigmoid.h"
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

const char *kernel_filename =
    "/home/shangar21/Documents/openCLNN/kernels/FC.cl";

std::string load_kernel(const char *filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file");
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

int main() {
  const int batch_size = 2;
  const int in_features = 3;
  const int out_features = 2;

  std::vector<float> X = {1, 2, 3, 4, 5, 6}; // Shape: (3,2)
  std::vector<float> gts = {1, 0, 0, 1};

  Module module;

  FC layer1(in_features, out_features, batch_size);
  module.addLayer(std::make_shared<FC>(layer1));

  Sigmoid layer2(out_features, batch_size);
  module.addLayer(std::make_shared<Sigmoid>(layer2));

  MSE lossFunc(out_features, batch_size);
  module.setLoss(std::make_shared<MSE>(lossFunc));

  module.forward(X);

  std::cout << "Output Matrix Y:\n";
  for (auto &i : module.getOutput()) {
    std::cout << i << std::endl;
  }

  module.loss(gts);

  std::cout << "Loss Value Y:\n";
  for (auto &i : module.lossVals) {
    std::cout << i << std::endl;
  }

  // Cleanup
  return 0;
}
