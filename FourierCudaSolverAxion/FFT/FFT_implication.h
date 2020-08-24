#pragma once

#include "stdafx.h"



class cuFFT
{
public:
	cuFFT(cudaStream_t _stream = cudaStreamDefault);
	cuFFT(const int dim, const int *_n, const int _BATCH = 1, cudaStream_t _stream = cudaStreamDefault);
	~cuFFT();

	void reset(const int dim, const int *_n, double _L, const int _BATCH = 1, cudaStream_t _stream = cudaStreamDefault);
	
	void forward(cudaCVector3 &f, cudaCVector3 &F, bool isNormed = true);
	void forward(cudaRVector3 &f, cudaCVector3 &F, bool isNormed = true);
	void inverce(cudaCVector3 &F, cudaCVector3 &f, bool isNormed = true);
	void inverce(cudaCVector3 &F, cudaRVector3 &f, bool isNormed = true);

	void setStream(cudaStream_t stream) {
		checkCudaErrors(cufftSetStream(planZ2ZF, stream));
		checkCudaErrors(cufftSetStream(planZ2ZI, stream));
		checkCudaErrors(cufftSetStream(planD2Z, stream));
		checkCudaErrors(cufftSetStream(planZ2D, stream));
	}

private:
	int dim;
	int *n;
	int BATCH;
	cufftHandle planZ2ZF, planZ2ZI, planD2Z, planZ2D;
	double L;
	int N;

	cufftCallbackStoreZ h_callbackForwardNormZ;
	cufftCallbackStoreD h_callbackForwardNormD;
	cufftCallbackStoreZ h_callbackInverseNormZ;
	cufftCallbackStoreD h_callbackInverseNormD;

	double* callbackData;

	cudaStream_t stream;
};
