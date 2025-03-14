#include "KernelUtils.h"

std::string readKernelFile(const std::string &filename) {
  std::string path = std::string(KERNELS_DIR) + "/" + filename;
  std::ifstream file;
  file.open(path);
  std::stringstream ss;
  ss << file.rdbuf();
  file.close();
  return ss.str();
}

void checkErr(const cl::Error &err, const char *operation) {
  std::cerr << "Error during operation '" << operation << "': " << err.what()
            << " (" << err.err() << ")" << std::endl;
  exit(1);
}
