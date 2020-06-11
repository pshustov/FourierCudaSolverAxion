#pragma once

#include "stdafx.h"

template <typename T> T reductionSum(int size, T* inData, cudaStream_t stream);
template <typename T> T reductionMax(int size, T* inData, cudaStream_t stream);