#pragma once

#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifndef KERNELS_DIR
#define KERNELS_DIR "./"
#endif

std::string readKernelFile(const std::string &filename);
void checkErr(const cl::Error &err, const char *operation);
