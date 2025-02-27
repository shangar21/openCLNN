#define CL_TARGET_OPENCL_VERSION 220

#include "FC.h"
#include "KernelUtils.h"
#include "Module.h"
#include <CL/cl.h>
#include <fstream>
#include <iostream>
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

  std::vector<float> X = {1, 2, 3, 4, 5, 6};             // Shape: (3,2)
  std::vector<float> W = {1, 0, 0, 1, 1, 1};             // Shape: (2,3)

  Module module;
  FC layer1(in_features, out_features, batch_size);
  std::cout << layer1 << std::endl;
  module.addLayer(std::move(layer1));
	FC layer2(out_features, 1, batch_size);
	std::cout << layer2 << std::endl;
	module.addLayer(std::move(layer2));
  module.forward(X);
  std::cout << "Output Matrix Y:\n";
	for (auto& i : module.Y){
		std::cout << i << std::endl;
	}
  

  // Cleanup
  return 0;
}
