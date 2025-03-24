#pragma once

#include "Layer.h"

class MSE : public Layer {
public:
  MSE(int in, int batch_size = 1);
  ~MSE();
  void setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                    const cl::Buffer &Y_buf) override;
  void setBackwardsKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                             cl::Context ctx) override;
};
