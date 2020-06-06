#include "stdafx.h"

__global__ void kernelForwardNorm(const size_t size, const size_t N, const double L, double *V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] * L / N;
	}
}
__global__ void kernelForwardNorm(const size_t size, const size_t N, const double L, complex *V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] * L / N;
	}
}
__global__ void kernelInverseNorm(const size_t size, const size_t N, const double L, double* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] / L;
	}
}
__global__ void kernelInverseNorm(const size_t size, const size_t N, const double L, complex* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] / L;
	}
}

void cuFFT::forward(cudaCVector &f, cudaCVector &F)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)f.getArray(), (cufftDoubleComplex*)F.getArray(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
	
	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)F.getN() / (double)BLOCK_SIZE));
	kernelForwardNorm<<<grid, block>>>(F.getN(), N, L, F.getArray());
	cudaDeviceSynchronize();
}
void cuFFT::forward(cudaRVector &f, cudaCVector &F)
{
	if (cufftExecD2Z(planD2Z, (cufftDoubleReal*)f.getArray(), (cufftDoubleComplex*)F.getArray()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
	
	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)F.getN() / (double)BLOCK_SIZE));
	kernelForwardNorm<<<grid, block>>>(F.getN(), N, L, F.getArray());
	cudaDeviceSynchronize();
}
void cuFFT::forward(cudaCVector3 &f, cudaCVector3 &F)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)f.getArray(), (cufftDoubleComplex*)F.getArray(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecZ2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
	
	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)F.size() / (double)BLOCK_SIZE));
	kernelForwardNorm<<<grid, block>>>(F.size(), N, L, F.getArray());
	cudaDeviceSynchronize();
}
void cuFFT::forward(cudaRVector3 &f, cudaCVector3 &F)
{
	if (cufftExecD2Z(planD2Z, (cufftDoubleReal*)f.getArray(), (cufftDoubleComplex*)F.getArray()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecD2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
	
	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)F.size() / (double)BLOCK_SIZE));
	kernelForwardNorm<<<grid, block>>>(F.size(), N, L, F.getArray());
	cudaDeviceSynchronize();
}

void cuFFT::inverce(cudaCVector &F, cudaCVector &f)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)F.getArray(), (cufftDoubleComplex*)f.getArray(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.getN() / (double)BLOCK_SIZE));
	kernelInverseNorm<<<grid, block>>>(f.getN(), N, L, f.getArray());
	cudaDeviceSynchronize();
}
void cuFFT::inverce(cudaCVector &F, cudaRVector &f)
{
	if (cufftExecZ2D(planZ2D, (cufftDoubleComplex*)F.getArray(), (cufftDoubleReal*)f.getArray()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.getN() / (double)BLOCK_SIZE));
	kernelInverseNorm<<<grid, block>>>(f.getN(), N, L, f.getArray());
	cudaDeviceSynchronize();
}
void cuFFT::inverce(cudaCVector3 &F, cudaCVector3 &f)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)F.getArray(), (cufftDoubleComplex*)f.getArray(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.size() / (double)BLOCK_SIZE));
	kernelInverseNorm<<<grid, block>>>(f.size(), N, L, f.getArray());
	cudaDeviceSynchronize();
}
void cuFFT::inverce(cudaCVector3 &F, cudaRVector3 &f)
{
	if (cufftExecZ2D(planZ2D, (cufftDoubleComplex*)F.getArray(), (cufftDoubleReal*)f.getArray()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.size() / (double)BLOCK_SIZE));
	kernelInverseNorm<<<grid, block>>>(f.size(), N, L, f.getArray());
	cudaDeviceSynchronize();
}

cuFFT::cuFFT(const int _dim, const int *_n, const int _BATCH) : dim(_dim), BATCH(_BATCH)
{
	n = new int[dim];
	for (size_t i = 0; i < dim; i++)
		n[i] = _n[i];

	int NX, NY, NZ;
	L = 1;

	switch (dim)
	{
	case 1:
		NX = n[0];
		N = NX;

		if (cufftPlan1d(&planZ2Z, NX, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan1d(&planD2Z, NX, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan1d(&planZ2D, NX, CUFFT_Z2D, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		break;

	case 3:
		NX = n[0];
		NY = n[1];
		NZ = n[2];
		N = NX * NY * NZ;

		if (cufftPlan3d(&planZ2Z, NX, NY, NZ, CUFFT_Z2Z) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan3d(&planD2Z, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan3d(&planZ2D, NX, NY, NZ, CUFFT_Z2D) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}

		break;
	
	default:
		throw;
	}
}
cuFFT::~cuFFT()
{
	cufftDestroy(planD2Z);
	cufftDestroy(planZ2D);
	cufftDestroy(planZ2Z);
	delete[] n;
}
void cuFFT::reset(const int _dim, const int *_n, double _L, const int _BATCH)
{
	dim = _dim;
	BATCH = _BATCH;
	L = _L;

	cufftDestroy(planD2Z);
	cufftDestroy(planZ2D);
	cufftDestroy(planZ2Z);
	delete[] n;
	n = new int[dim];
	for (size_t i = 0; i < dim; i++)
		n[i] = _n[i];

	int NX, NY, NZ;

	switch (dim)
	{
	case 1:
		NX = n[0];
		N = NX;

		if (cufftPlan1d(&planZ2Z, NX, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan1d(&planD2Z, NX, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan1d(&planZ2D, NX, CUFFT_Z2D, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		break;

	case 3:
		NX = n[0];
		NY = n[1];
		NZ = n[2];
		N = NX * NY * NZ;

		if (cufftPlan3d(&planZ2Z, NX, NY, NZ, CUFFT_Z2Z) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan3d(&planD2Z, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}
		if (cufftPlan3d(&planZ2D, NX, NY, NZ, CUFFT_Z2D) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			return;
		}

		break;

	default:
		throw;
	}
}