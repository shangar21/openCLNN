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

void Module::setLoss(std::shared_ptr<Layer> loss) { lossFunction = loss; }

void Module::forward(std::vector<float> X) {
  cl_mem X_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     X.size() * sizeof(float), X.data(), nullptr);

  for (std::shared_ptr<Layer> &layer : fullyConnectedLayers) {
    layer->getKernel(context, platform, device);
    layer->setKernelArg(X_buf, context);
    checkErr(clEnqueueNDRangeKernel(queue, layer->kernel, 2, nullptr,
                                    layer->launchConfig, nullptr, 0, nullptr,
                                    nullptr),
             "Enqueue Kernel");
    clFinish(queue);

    clRetainMemObject(layer->Y_buf);

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

void Module::loss(std::vector<float> gts) {
  Y = getOutput();
  cl_int err;
  cl_mem gt_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     gts.size() * sizeof(float), gts.data(), &err);
  checkErr(err, "Set GT buffer");
  cl_mem X_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     Y.size() * sizeof(float), Y.data(), &err);
  checkErr(err, "Set Output buffer");
  lossFunction->getKernel(context, platform, device);
  lossFunction->setKernelArg(X_buf, context, gt_buf);
  checkErr(clEnqueueNDRangeKernel(queue, lossFunction->kernel, 2, nullptr,
                                  lossFunction->launchConfig, nullptr, 0,
                                  nullptr, nullptr),
           "Enqeue Loss Kernel");
  int outSize = lossFunction->batch_size * lossFunction->out_features;
  lossVals.clear();
  lossVals.resize(outSize);
  checkErr(clEnqueueReadBuffer(queue, lossFunction->Y_buf, CL_TRUE, 0,
                               outSize * sizeof(float), lossVals.data(), 0,
                               nullptr, nullptr),
           "Read Loss Output Buffer");
  if (X_buf)
    clReleaseMemObject(X_buf);
  if (gt_buf)
    clReleaseMemObject(gt_buf);
}

void Module::backwards() {
  // Getting loss gradient before propogating it back
  cl_int err;
  cl_mem dLoss_buf = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      fullyConnectedLayers[0]->batch_size * sizeof(float), nullptr, &err);
  checkErr(err, "Set Loss Grad Buffer");
  lossFunction->getKernel(context, platform, device, true);
  lossFunction->setBackwardsKernelArg(dLoss_buf, context);
  checkErr(clEnqueueNDRangeKernel(queue, lossFunction->kernel, 2, nullptr,
                                  lossFunction->launchConfig, nullptr, 0,
                                  nullptr, nullptr),
           "Enqueue Loss Kernel");
  clFinish(queue);

  for (std::shared_ptr<Layer> &layer :
       std::views::reverse(fullyConnectedLayers)) {
  }
}

std::vector<float> Module::getOutput() {
  std::vector<float> out;
  std::shared_ptr<Layer> lastLayer = fullyConnectedLayers.back();
  int outSize = lastLayer->batch_size * lastLayer->out_features;
  out.resize(outSize);
  checkErr(clEnqueueReadBuffer(queue, lastLayer->Y_buf, CL_TRUE, 0,
                               outSize * sizeof(float), out.data(), 0, nullptr,
                               nullptr),
           "Read Output Buffer");
  clFinish(queue);
  return out;
}
