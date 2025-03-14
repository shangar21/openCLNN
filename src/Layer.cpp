#include "Layer.h"

cl_program Layer::getProgram(cl_context ctx, bool backwards) {
  std::string s = backwards ? backwardsSource : source;
  const char *code = s.c_str();
  size_t size = s.size();
  return clCreateProgramWithSource(ctx, 1, &code, &size, nullptr);
}

cl_kernel Layer::getKernel(cl_context ctx, cl_platform_id, cl_device_id device,
                           bool backwards) {
  cl_kernel &setKernel = backwards ? backwardsKernel : kernel;

  if (setKernel != nullptr)
    return setKernel;

  program = getProgram(ctx);
  cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  checkErr(err, "Failed to build program");

  setKernel = clCreateKernel(
      program, backwards ? backwardsName.c_str() : name.c_str(), &err);
  checkErr(err, "Failed to create kernel");

  Y_buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                     out_features * batch_size * sizeof(float), nullptr, &err);
  checkErr(err, "Failed to create Y_buf");
  return setKernel;
}
