#pragma once

#include "stdafx.h"

template <typename T> class vector;
template <typename T> class vector3;
template <typename T> class d_cudaVector;

template <typename T>
class cudaVector
{
public:
	explicit cudaVector(const size_t _N = 1) : N(_N)
	{
		cudaMalloc(&Array, N * sizeof(T));
	}
	cudaVector(const cudaVector& _V) : N(_V.get_N())
	{
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	cudaVector& operator=(const cudaVector& _V)
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
	~cudaVector()
	{
		cudaFree(Array);
	}
	
	cudaVector(const vector<T>& _V) : N(_V.get_N())
	{
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyHostToDevice);
	}
	cudaVector& operator=(const vector<T>& _V)
	{
		N = _V.get_N();

		cudaFree(Array);
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyHostToDevice);

		return *this;
	}

	size_t get_N() const { return N; }
	T* get_Array() const { return Array; }

	void set_size_erase(const size_t _N)
	{
		N = _N;
		cudaFree(Array);
		cudaMalloc(&Array, N * sizeof(T));
	}

	friend class vector<T>;
	friend class d_cudaVector<T>;


private:
	size_t N;
	T* Array;
};

using cudaRVector = cudaVector<double>;
using cudaCVector = cudaVector<complex>;




template <typename T>
class d_cudaVector
{
public:
	d_cudaVector(const cudaVector<T>& _V)// :N(_V.get_N())
	{
		Array = _V.Array;
		std::cout << "Adress of Array: " << Array << std::endl;
		std::cout << "Adress of Class: " << this << std::endl;
	}
	~d_cudaVector() 
	{
	}

	__device__ size_t get_N() const { return N; }
	__device__ T& operator() (size_t i) { return Array[i]; }
	__device__ const T& operator() (size_t i) const { return Array[i]; }

	friend class cudaVector<T>;

private:
	T* Array;
};

using d_cudaRVector = d_cudaVector<double>;
using d_cudaCVector = d_cudaVector<complex>;













template <typename T>
class cudaVector3
{
public:

	__host__ explicit cudaVector3(const size_t _N1 = 1, const size_t _N2 = 1, const size_t _N3 = 1) : N1(_N1), N2(_N2), N3(_N3)
	{
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
	}
	__host__ cudaVector3(const cudaVector3& _V) : N1(_V.get_N1()), N2(_V.get_N2()), N3(_V.get_N3())
	{
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	__host__ cudaVector3& operator=(const cudaVector3& _V)
	{
		if (this != &_V)
		{
			N1 = _V.get_N1();
			N2 = _V.get_N2();
			N3 = _V.get_N3();

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

	__host__ cudaVector3(const vector3<T>& _V) : N1(_V.get_N1()), N2(_V.get_N2()), N3(_V.get_N3())
	{
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ cudaVector3& operator=(const vector3<T>& _V)
	{
		N1 = _V.get_N1();
		N2 = _V.get_N2();
		N3 = _V.get_N3();

		cudaFree(Array);
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyHostToDevice);
		
		return *this;
	}

	__host__ size_t get_N1() const { return N1; }
	__host__ size_t get_N2() const { return N2; }
	__host__ size_t get_N3() const { return N3; }
	__host__ size_t size() const { return N1*N2*N3; }
	

	T* get_Array() const { return Array; }
	__host__ void set_size_erase(const size_t _N1, const size_t _N2, const size_t _N3)
	{
		cudaFree(Array);
		N1 = _N1;
		N2 = _N2;
		N3 = _N3;
		cudaMalloc(&Array, N1*N2*N3 * sizeof(T));
	}
	
	friend class vector3<T>;

private:
	size_t N1, N2, N3;
	T* Array;
};

using cudaRVector3 = cudaVector3<double>;
using cudaCVector3 = cudaVector3<complex>;