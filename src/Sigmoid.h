#pragma once

#include "Layer.h"

class Sigmoid : public Layer {
public:
  Sigmoid(int in, int batch_size = 1);
  ~Sigmoid();
  void setKernelArg(cl_mem &X_buf, cl_context ctx,
                    const cl_mem &gt_buf = nullptr) override;
};
