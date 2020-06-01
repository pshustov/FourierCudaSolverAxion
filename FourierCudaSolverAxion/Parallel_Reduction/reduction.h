#pragma once

#include "stdafx.h"

#define MAXIMUM 0
#define SUMMATION 1
#define MEAN 2
#define SIGMA2 3
#define SIGMA4 4

//#ifndef MIN
//#define MIN(x,y) ((x < y) ? x : y)
//#endif
//
//#ifndef MAX
//#define MAX(x,y) ((x > y) ? x : y)
//#endif

template <typename T>
T reductionSum(int size, T* inData);