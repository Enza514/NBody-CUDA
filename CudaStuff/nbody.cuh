#pragma once

#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "defines.h"
#include "helper_cuda.h"

void launchKernel(float3*, float3*, float*);
__global__ void kernel(float3*, float3*, float*);

#endif
