#include "Module.h"

Module::Module() {
  checkErr(clGetPlatformIDs(1, &platform, nullptr), "Get Platform");
  checkErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
           "Get Device");
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);
}

Module::~Module() {}

void Module::addLayer(std::shared_ptr<Layer> layer) {
  if (fullyConnectedLayers.size() == 0)
    inputSize = layer->in_features;
  fullyConnectedLayers.push_back(layer);
}

void Module::forward(std::vector<float> X) {
  cl_mem X_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     X.size() * sizeof(float), X.data(), nullptr);

  for (std::shared_ptr<Layer> &layer : fullyConnectedLayers) {
    std::cout << layer->name << std::endl;
    size_t global_work_size[2] = {layer->batch_size, layer->out_features};
    layer->getKernel(context, platform, device);
    layer->setKernelArg(X_buf, context);

    checkErr(clEnqueueNDRangeKernel(queue, layer->kernel, 2, nullptr,
                                    global_work_size, nullptr, 0, nullptr,
                                    nullptr),
             "Enqueue Kernel");
    clFinish(queue);

    int outSize = layer->batch_size * layer->out_features;
    Y.clear();
    Y.resize(outSize);
    checkErr(clEnqueueReadBuffer(queue, layer->Y_buf, CL_TRUE, 0,
                                 outSize * sizeof(float), Y.data(), 0, nullptr,
                                 nullptr),
             "Read Output Buffer");

    if (X_buf != layer->Y_buf) {
      if (X_buf)
        clReleaseMemObject(X_buf);
      X_buf = layer->Y_buf;
      clRetainMemObject(X_buf);
    }
  }

  if (X_buf)
    clReleaseMemObject(X_buf);
}

std::vector<float> Module::getOutput() {
  std::vector<float> out;
  std::shared_ptr<Layer> lastLayer = fullyConnectedLayers.back();
  int outSize = lastLayer->batch_size * lastLayer->out_features;
  checkErr(clEnqueueReadBuffer(queue, Y_buf, CL_TRUE, 0,
                               outSize * sizeof(float), out.data(), 0, nullptr,
                               nullptr),
           "Read Output Buffer");
  return out;
}
