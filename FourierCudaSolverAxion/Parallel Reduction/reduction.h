#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <cooperative_groups.h>
#include <ctime>
#include <iostream>
#include <vector>

#define MAXIMUM 0
#define SUMMATION 1
#define MEAN 2
#define SIGMA2 3
#define SIGMA4 4

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif
