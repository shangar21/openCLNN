#pragma once

#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifndef KERNELS_DIR
#define KERNELS_DIR "./"
#endif

std::string readKernelFile(const std::string &filename);
void checkErr(cl_int err, const char *operation);
