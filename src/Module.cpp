#include "Module.h"

Module::Module() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    std::cout << "Platform Search Error!\n";
    exit(1);
  }
  platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.empty()) {
    std::cout << "Device Search Error!\n";
    exit(1);
  }
  device = devices.front();

  context = cl::Context(device);
  queue = cl::CommandQueue(context, device);
}

Module::~Module() {}

void Module::addLayer(std::shared_ptr<Layer> layer) {
  if (fullyConnectedLayers.size() == 0)
    inputSize = layer->in_features;
  fullyConnectedLayers.push_back(layer);
}

void Module::setLoss(std::shared_ptr<Layer> loss) { lossFunction = loss; }

void Module::forward(std::vector<float> X) {
  cl::Buffer X_buf =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 X.size() * sizeof(float), X.data());

  for (std::shared_ptr<Layer> &layer : fullyConnectedLayers) {
    layer->getKernel(context, platform, device);
    layer->setKernelArg(X_buf, context);
    queue.enqueueNDRangeKernel(
        layer->kernel, cl::NullRange,
        cl::NDRange(layer->launchConfig[0], layer->launchConfig[1]));
    queue.finish();

    if (X_buf != layer->Y_buf)
      X_buf = layer->Y_buf;
  }
}

void Module::loss(std::vector<float> gts) {
  Y = getOutput();
  cl::Buffer gt_buf =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 gts.size() * sizeof(float), gts.data());
  cl::Buffer X_buf =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 Y.size() * sizeof(float), Y.data());
  lossFunction->getKernel(context, platform, device);
  lossFunction->setKernelArg(X_buf, context, gt_buf);
  queue.enqueueNDRangeKernel(lossFunction->kernel, cl::NullRange,
                             cl::NDRange(lossFunction->launchConfig[0],
                                         lossFunction->launchConfig[1]));
  int outSize = lossFunction->batch_size * lossFunction->out_features;
  lossVals.clear();
  lossVals.resize(outSize);
  queue.enqueueReadBuffer(lossFunction->Y_buf, CL_TRUE, 0,
                          outSize * sizeof(float), lossVals.data());
}

void Module::backwards() {
  cl::Buffer dLoss_buf(context, CL_MEM_READ_WRITE,
                       fullyConnectedLayers[0]->batch_size * sizeof(float));
  cl::Buffer dX_buf(context, CL_MEM_READ_WRITE,
                    fullyConnectedLayers[0]->batch_size * sizeof(float));
  lossFunction->getKernel(context, platform, device, true);
  lossFunction->setBackwardsKernelArg(dLoss_buf, dX_buf, context);
  queue.enqueueNDRangeKernel(lossFunction->kernel, cl::NullRange,
                             cl::NDRange(lossFunction->launchConfig[0],
                                         lossFunction->launchConfig[1]));
  queue.finish();

  for (auto it = fullyConnectedLayers.rbegin();
       it != fullyConnectedLayers.rend(); ++it) {
    std::shared_ptr<Layer> layer = *it;
    layer->getKernel(context, platform, device, true);
    layer->setBackwardsKernelArg(dLoss_buf, dX_buf, context);
    queue.enqueueNDRangeKernel(layer->kernel, cl::NullRange,
                               cl::NDRange(layer->backwardsLaunchConfig[0],
                                           layer->backwardsLaunchConfig[1]));
    queue.finish();
    if (dLoss_buf != layer->dY_buf)
      dLoss_buf = layer->dY_buf;
    if (dX_buf != layer->dX_buf)
      dX_buf = layer->dX_buf;
  }
}

std::vector<float> Module::getOutput() {
  std::vector<float> out;
  std::shared_ptr<Layer> lastLayer = fullyConnectedLayers.back();
  int outSize = lastLayer->batch_size * lastLayer->out_features;
  out.resize(outSize);
  queue.enqueueReadBuffer(lastLayer->Y_buf, CL_TRUE, 0, outSize * sizeof(float),
                          out.data());
  queue.finish();
  return out;
}
