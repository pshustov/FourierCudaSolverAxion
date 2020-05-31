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
		L = 1;
		N = 1;
		BATCH = 1;

		cufftPlan1d(&planZ2Z, N, CUFFT_Z2Z, BATCH);	
		cufftPlan1d(&planD2Z, N, CUFFT_D2Z, BATCH);
		cufftPlan1d(&planZ2D, N, CUFFT_Z2D, BATCH);

	}
	cuFFT(const int dim, const int *_n, const int _BATCH = 1);
	~cuFFT();

	void reset(const int dim, const int *_n, double _L, const int _BATCH = 1);

	void forward(cudaCVector &f, cudaCVector &F);
	void forward(cudaRVector &f, cudaCVector &F);
	void inverce(cudaCVector &F, cudaCVector &f);
	void inverce(cudaCVector &F, cudaRVector &f);
	
	void forward(cudaCVector3 &f, cudaCVector3 &F);
	void forward(cudaRVector3 &f, cudaCVector3 &F);
	void inverce(cudaCVector3 &F, cudaCVector3 &f);
	void inverce(cudaCVector3 &F, cudaRVector3 &f);


private:
	int dim;
	int *n;
	int BATCH;
	cufftHandle planZ2Z, planD2Z, planZ2D;
	double L;
	int N;
};