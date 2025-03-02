#pragma once

#include "KernelUtils.h"
#include <CL/cl.h>
#include <vector>

class Layer {
	public:
		std::string source = "";
		std::string name = "";
		int in_features = 0;
		int out_features = 0;
		int batch_size = 1;
		cl_kernel kernel = nullptr;
		cl_mem Y_buf = nullptr;
		cl_mem dY_buf = nullptr;
		cl_program program = nullptr;

		virtual ~Layer() = default;
		virtual void setKernelArg(cl_mem &X_buf, cl_context ctx) = 0;
		cl_kernel getKernel(cl_context ctx, cl_platform_id platform, cl_device_id device);
		cl_program getProgram(cl_context ctx);
};
