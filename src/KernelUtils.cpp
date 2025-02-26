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

void checkErr(cl_int err, const char *operation) {
  if (err != CL_SUCCESS) {
    std::cerr << "Error: " << operation << " (" << err << ")" << std::endl;
    exit(1);
  }
}
