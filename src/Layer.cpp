#include "Layer.h"

cl_program Layer::getProgram(cl_context ctx) {
  const char *code = source.c_str();
  size_t size = source.size();
  return clCreateProgramWithSource(ctx, 1, &code, &size, nullptr);
}

cl_kernel Layer::getKernel(cl_context ctx, cl_platform_id, cl_device_id device) {
  program = getProgram(ctx);
  cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  checkErr(err, "Failed to build program");

  kernel = clCreateKernel(program, name.c_str(), &err);
  checkErr(err, "Failed to create kernel");

  Y_buf =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                     out_features * batch_size * sizeof(float), nullptr, &err);
  checkErr(err, "Failed to create Y_buf");
  return kernel;
}
