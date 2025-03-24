#pragma once

#include "Module.h"
#include <vector>

class Optimizer {
public:
  Module &module;
  Optimizer(Module &m) : module(m){};
  virtual void step() = 0;
};
