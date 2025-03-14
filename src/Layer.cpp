#include "Layer.h"

cl::Program Layer::getProgram(cl::Context ctx, bool backwards) {
  std::string s = backwards ? backwardsSource : source;
  cl::Program::Sources sources;
  sources.push_back({s.c_str(), s.length()});
  return cl::Program(ctx, sources);
}

cl::Kernel Layer::getKernel(cl::Context ctx, cl::Platform, cl::Device device,
                            bool backwards) {
  cl::Kernel &setKernel = backwards ? backwardsKernel : kernel;

  if (setKernel())
    return setKernel;

  program = getProgram(ctx);
  try {
    program.build({device});
  } catch (const cl::Error &err) {
    checkErr(err, "Failed to build program");
  }

  setKernel =
      cl::Kernel(program, backwards ? backwardsName.c_str() : name.c_str());

  Y_buf = cl::Buffer(ctx, CL_MEM_READ_WRITE,
                     out_features * batch_size * sizeof(float));
  return setKernel;
}
