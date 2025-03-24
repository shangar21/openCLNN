#include "Layer.h"

cl::Program Layer::getProgram(cl::Context ctx, std::string &source) {
  cl::Program::Sources sources;
  sources.push_back({source.c_str(), source.length()});
  return cl::Program(ctx, sources);
}

cl::Kernel Layer::getKernel(cl::Context ctx, cl::Platform platform,
                            cl::Device device, std::string &source,
                            std::string &kernelName, cl::Kernel &setKernel) {

  if (setKernel())
    return setKernel;

  program = getProgram(ctx, source);
  try {
    program.build(device);
  } catch (cl::Error &err) {
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::cerr << "Build log for device " << device.getInfo<CL_DEVICE_NAME>()
              << ":\n"
              << buildLog << std::endl;
    throw err;
  }

  setKernel = cl::Kernel(program, name.c_str());

  Y_buf = cl::Buffer(ctx, CL_MEM_READ_WRITE,
                     out_features * batch_size * sizeof(float));
  return setKernel;
}

cl::Kernel Layer::getForwardKernel(cl::Context ctx, cl::Platform platform,
                                   cl::Device device) {
  return getKernel(ctx, platform, device, source, name, kernel);
}

cl::Kernel Layer::getBackwardsInputKernel(cl::Context ctx,
                                          cl::Platform platform,
                                          cl::Device device) {
  return getKernel(ctx, platform, device, backwardsInputSource,
                   backwardsInputName, backwardsInputKernel);
}

cl::Kernel Layer::getBackwardsWeightKernel(cl::Context ctx,
                                           cl::Platform platform,
                                           cl::Device device) {
  return getKernel(ctx, platform, device, backwardsWeightSource,
                   backwardsWeightName, backwardsWeightKernel);
}
