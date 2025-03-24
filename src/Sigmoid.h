#pragma once

#include "Layer.h"

class Sigmoid : public Layer {
public:
  Sigmoid(int in, int batch_size = 1);
  ~Sigmoid();
  void setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                    const cl::Buffer &gt_buf = cl::Buffer()) override;
  void setBackwardsKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                             cl::Context ctx) override;
};
