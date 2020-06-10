#pragma once

#include "stdafx.h"

template <typename T> class vector;
template <typename T> class cudaVectorDev;

/// 1D cuda Vector for host
template <typename T>
 class cudaVector
{
public:
	__host__ explicit cudaVector(const size_t _N = 1) : N(_N)
	{
		cudaMalloc(&Array, N * sizeof(T));
	}
	__host__ cudaVector(const cudaVector& _V) : N(_V.getN())
	{
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToDevice);
		
	}
	__host__ cudaVector& operator=(const cudaVector& _V)
	{
		if (this != &_V)
		{
			N = _V.N;
			
			cudaFree(Array);
			cudaMalloc(&Array, N * sizeof(T));
			cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToDevice);
		}
		return *this;
	}
	__host__ ~cudaVector()
	{
		cudaFree(Array);
	}
	
	__host__ cudaVector(const vector<T>& _V) : N(_V.getN())
	{
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ cudaVector& operator=(const vector<T>& _V)
	{
		N = _V.getN();

		cudaFree(Array);
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyHostToDevice);

		return *this;
	}

	__host__ size_t getN() const { return N; }
	
	__host__ void set(const size_t _N)
	{
		N = _N;
		cudaFree(Array);
		cudaMalloc(&Array, N * sizeof(T));
	}
	
	__host__ T* getArray() { return Array; }
	
	__host__ T getSum() { return reductionSum<T>(N, Array); }

	friend class vector<T>;
	friend class cudaVectorDev<T>;


private:
	size_t N;
	T* Array;
};

using cudaRVector = cudaVector<double>;
using cudaCVector = cudaVector<complex>;

/// 1D cuda Vector for Device
template <typename T>
class cudaVectorDev
{
public:
	__host__ cudaVectorDev(const cudaVector<T>& _V) :N(_V.getN())
	{
		Array = _V.Array;
	}
	__host__ ~cudaVectorDev() { }

	__device__ size_t getN() const { return N; }
	__device__ T& operator() (size_t i) { return Array[i]; }
	__device__ const T& operator() (size_t i) const { return Array[i]; }

	friend class cudaVector<T>;

private:
	size_t N;
	T* Array;
};

using cudaRVectorDev = cudaVectorDev<double>;
using cudaCVectorDev = cudaVectorDev<complex>;

template <typename T> class vector3;
template <typename T> class cudaVector3Dev;


/// 3D cuda Vector for host
template <typename T>
class cudaVector3
{
public:

	__host__ explicit cudaVector3(const size_t _N1 = 1, const size_t _N2 = 1, const size_t _N3 = 1) : N1(_N1), N2(_N2), N3(_N3)
	{
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
	}
	__host__ cudaVector3(const cudaVector3& _V) : N1(_V.getN1()), N2(_V.getN2()), N3(_V.getN3())
	{
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	__host__ cudaVector3& operator=(const cudaVector3& _V)
	{
		if (this != &_V)
		{
			N1 = _V.getN1();
			N2 = _V.getN2();
			N3 = _V.getN3();

			cudaFree(Array);
			cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
			cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyDeviceToDevice);
		}
		return *this;
	}
	__host__ ~cudaVector3()
	{
		cudaFree(Array);
	}

	__host__ cudaVector3(const vector3<T>& _V) : N1(_V.getN1()), N2(_V.getN2()), N3(_V.getN3())
	{
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ cudaVector3& operator=(const vector3<T>& _V)
	{
		N1 = _V.getN1();
		N2 = _V.getN2();
		N3 = _V.getN3();

		cudaFree(Array);
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyHostToDevice);
		
		return *this;
	}

	__host__ size_t getN1() const { return N1; }
	__host__ size_t getN2() const { return N2; }
	__host__ size_t getN3() const { return N3; }
	__host__ size_t size() const { return N1*N2*N3; }

	__host__ void set(const size_t _N1, const size_t _N2, const size_t _N3)
	{
		cudaFree(Array);
		N1 = _N1;
		N2 = _N2;
		N3 = _N3;
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
	}
	
	__host__ T* getArray() { return Array; }
	
	__host__ T getSum(cudaStream_t& stream) { return reductionSum<T>(size(), Array); }
	
	friend class vector3<T>;
	friend class cudaVector3Dev<T>;

private:
	size_t N1, N2, N3;
	T* Array;
};

using cudaRVector3 = cudaVector3<double>;
using cudaCVector3 = cudaVector3<complex>;

/// 3D cuda Vector for Device
template <typename T>
class cudaVector3Dev
{
public:
	__host__ cudaVector3Dev(const cudaVector3<T>& _V) : N1(_V.getN1()), N2(_V.getN2()), N3(_V.getN3())
	{
		Array = _V.Array;
	}
	__host__ ~cudaVector3Dev() {}

	__device__ size_t size() const { return N1 * N2 * N3; }
	__device__ size_t getN1() const { return N1; }
	__device__ size_t getN2() const { return N2; }
	__device__ size_t getN3() const { return N3; }

	__device__ T& operator() (size_t i) { return Array[i]; }
	__device__ const T& operator() (size_t i) const { return Array[i]; }

	__device__ T& operator() (size_t i, size_t j, size_t k) { return Array[(i * N2 + j) * N3 + k]; }
	__device__ const T& operator() (size_t i, size_t j, size_t k) const { return Array[(i * N2 + j) * N3 + k]; }

	friend class cudaVector3<T>;

private:
	size_t N1, N2, N3;
	T* Array;
};

using cudaRVector3Dev = cudaVector3Dev<double>;
using cudaCVector3Dev = cudaVector3Dev<complex>;