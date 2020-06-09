#pragma once

#include "stdafx.h"

void cufft(cudaCVector &f, cudaCVector &F);
void cufft(cudaRVector &f, cudaCVector &F);
void cuifft(cudaCVector &F, cudaCVector &f);
void cuifft(cudaCVector &F, cudaRVector &f);

void cufft(cudaCVector3 &f, cudaCVector3 &F);
void cufft(cudaRVector3 &f, cudaCVector3 &F);
void cuifft(cudaCVector3 &F, cudaCVector3 &f);
void cuifft(cudaCVector3 &F, cudaRVector3 &f);

class cuFFT
{
public:
	cuFFT()
	{
		dim = 1;
		n = new int[dim];
		BATCH = 1;
		L = 0;
		N = 0;
	}
	cuFFT(const int dim, const int* _n, const int _BATCH = 1, cudaStream_t _stream = cudaStreamLegacy);
	~cuFFT();

	void reset(const int dim, const int *_n, double _L, const int _BATCH = 1, cudaStream_t _stream = cudaStreamLegacy);

	void forward(cudaCVector &f, cudaCVector &F);
	void forward(cudaRVector &f, cudaCVector &F);
	void inverce(cudaCVector &F, cudaCVector &f);
	void inverce(cudaCVector &F, cudaRVector &f);
	
	void forward(cudaCVector3 &f, cudaCVector3 &F, bool isNormed = true);
	void forward(cudaRVector3 &f, cudaCVector3 &F, bool isNormed = true);
	void inverce(cudaCVector3 &F, cudaCVector3 &f, bool isNormed = true);
	void inverce(cudaCVector3 &F, cudaRVector3 &f, bool isNormed = true);

	void setStreamAll(cudaStream_t _stream) {
		cufftSetStream(planZ2Z, _stream);
		cufftSetStream(planD2Z, _stream);
		cufftSetStream(planZ2D, _stream);
	}

private:
	int dim;
	int *n;
	int BATCH;
	cufftHandle planZ2Z, planD2Z, planZ2D;
	
	double L;
	int N;

	cudaStream_t stream;
};
