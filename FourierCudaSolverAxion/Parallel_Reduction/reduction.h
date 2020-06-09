#pragma once

#include "stdafx.h"

template <typename T> T reductionSum(int size, T* inData, cudaStream_t& stream);

//template<typename T>
//using func_t = T(*) (T, T);
//
//template <typename T>
//__host__ __device__ T funSum(T A, T B)
//{
//	return A + B;
//}
//
//template <typename T>
//__device__  func_t<T> p_funSum = funSum<T>;