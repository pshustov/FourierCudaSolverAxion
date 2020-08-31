#pragma once
#include <cufft.h>
#include <cufftXt.h>

class cuFFT
{
public:
	cuFFT(cudaStream_t _stream = cudaStreamDefault);
	cuFFT(const int dim, const int *_n, const int _BATCH = 1, cudaStream_t _stream = cudaStreamDefault);
	~cuFFT();

	void reset(const int dim, const int *_n, double _L, const int _BATCH = 1, cudaStream_t _stream = cudaStreamDefault);
	
	void forward(cudaCVector3 &f, cudaCVector3 &F);
	void forward(cudaRVector3 &f, cudaCVector3 &F);
	void inverce(cudaCVector3 &F, cudaCVector3 &f);
	void inverce(cudaCVector3 &F, cudaRVector3 &f);

	void setStream(cudaStream_t stream);

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