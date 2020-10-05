#pragma once
#include <cufft.h>
#include <cufftXt.h>

class cuFFT
{
public:
	cuFFT();
	cuFFT(const int _dim, const int *_n, real _volume, const int _BATCH = 1, cudaStream_t _stream = cudaStreamDefault);
	~cuFFT();

	void reset(const int _dim, const int *_n, real _volume, const int _BATCH = 1, cudaStream_t _stream = cudaStreamDefault);
	void clear();
	
	void forward(cudaCVector3 &f, cudaCVector3 &F);
	void forward(cudaRVector3 &f, cudaCVector3 &F);
	void inverce(cudaCVector3 &F, cudaCVector3 &f);
	void inverce(cudaCVector3 &F, cudaRVector3 &f);

	void setStream(cudaStream_t stream);

private:
	bool isInitialized;

	int dim;
	int *n;
	int BATCH;
	cufftHandle planC2CF, planC2CI, planR2C, planC2R;
	real volume;
	int N;

	cufftCallbackStoreC h_callbackForwardNormC;
	cufftCallbackStoreR h_callbackForwardNormR;
	cufftCallbackStoreC h_callbackInverseNormC;
	cufftCallbackStoreR h_callbackInverseNormR;

	cufftCallbackStoreZ h_callbackForwardNormZ;
	cufftCallbackStoreD h_callbackForwardNormD;
	cufftCallbackStoreZ h_callbackInverseNormZ;
	cufftCallbackStoreD h_callbackInverseNormD;

	real* callbackData;

	cudaStream_t stream;
};
