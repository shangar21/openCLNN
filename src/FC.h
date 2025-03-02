#pragma once

#include "Layer.h"

class FC : public Layer {
public:
  std::vector<float> W;
  cl_mem weightBuffer = nullptr;

  FC(int in, int out, int batch_size = 1, bool randomize = true);
  ~FC();
  void getWeightBuffers(cl_context ctx);
  friend std::ostream &operator<<(std::ostream &os, const FC &fc);
  void setKernelArg(cl_mem &X_buf, cl_context ctx,
                    const cl_mem &gt_buf = nullptr) override;
};
