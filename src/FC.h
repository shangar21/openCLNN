#pragma once

#include "Layer.h"

class FC : public Layer {
public:
  std::vector<float> W;
  cl::Buffer weightBuffer;

  FC(int in, int out, int batch_size = 1, bool randomize = true);
  ~FC();
  void getWeightBuffers(cl::Context ctx);
  friend std::ostream &operator<<(std::ostream &os, const FC &fc);
  void setKernelArg(cl::Buffer &X_buf, cl::Context ctx,
                    const cl::Buffer &gt_buf = cl::Buffer()) override;
  void setBackwardsWeightKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                                   cl::Context ctx) override;
  void setBackwardsInputKernelArg(cl::Buffer &dLoss_buf, cl::Buffer &dX_buf,
                                   cl::Context ctx) override;
};
