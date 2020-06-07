#pragma once

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <future>

#include <cufft.h>
#include <cooperative_groups.h>
//#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "complexCuda.h"
#include "reduction.h"
#include "vector.h"
#include "vectorCuda.h"
#include "FFT_implication.h"
#include "gridCuda.h"
#include "equations.h"
#include "distribution.h"
#include "system.h"

constexpr auto Ma_PI = 3.1415926535897932384626433832795;
constexpr auto BLOCK_SIZE = 128;
constexpr auto N_MIN = 32;

//__global__ void kernalStepSymplectic(const int N, const int s, const double dt, 
//	double *C, double *D, double *k_sqr, complex *Q, complex *P, complex *Nonlin);

__global__ void kernelDer(const int N, complex* T, double *k, complex *Q);
__global__ void kernel_Phi4_Phi6(const int N, double *t, double *q, const double lambda, const double g);
__global__ void kernelCulcRhoReal(const int N, double *rho, double *q, double *p, const double lambda, const double g);
__global__ void kernelAddMullSqr(const int N, double* S, double* A, double m);
__global__ void kernelSyncBuf(double *A, double *A0);
__global__ void kernelGetOmega(const int N, double *omega, double *kSqr, const double sigma2, const double sigma4, const double lambda, const double g);
