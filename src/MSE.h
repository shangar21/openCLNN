#pragma once

#include "Layer.h"

class MSE : public Layer {
public:
  MSE(int in, int batch_size = 1);
  ~MSE();
  void setKernelArg(cl_mem &X_buf, cl_context ctx,
                    const cl_mem &Y_buf) override;
};
