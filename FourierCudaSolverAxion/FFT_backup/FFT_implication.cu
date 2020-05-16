#include "stdafx.h"

__global__ void kernelNorm(int N, double *V)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		V[i] = V[i] / N;
	}
}
__global__ void kernelNorm(int N, complex *V)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		V[i] = V[i] / N;
	}
}

void cufft(cudaCVector &f, cudaCVector &F)
{
	int NX = (int)f.get_N();
	auto BATCH = 1;

	cufftHandle plan;
	if (cufftPlan1d(&plan, NX, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)f.get_Array(), (cufftDoubleComplex*)F.get_Array(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();
}
void cufft(cudaRVector &f, cudaCVector &F)
{
	int NX = (int)f.get_N();
	auto BATCH = 1;

	cufftHandle plan;
	if (cufftPlan1d(&plan, NX, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecD2Z(plan, (cufftDoubleReal*)f.get_Array(), (cufftDoubleComplex*)F.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();
}
void cufft(cudaCVector3 &f, cudaCVector3 &F)
{
	int NX = (int)f.get_N1();
	int NY = (int)f.get_N2();
	int NZ = (int)f.get_N3();

	cufftHandle plan;
	if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_Z2Z) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)f.get_Array(), (cufftDoubleComplex*)F.get_Array(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecZ2Z Forward failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();
}
void cufft(cudaRVector3 &f, cudaCVector3 &F)
{
	int NX = (int)f.get_N1();
	int NY = (int)f.get_N2();
	int NZ = (int)f.get_N3();

	cufftHandle plan;
	if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecD2Z(plan, (cufftDoubleReal*)f.get_Array(), (cufftDoubleComplex*)F.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecD2Z Forward failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();
}

void cuifft(cudaCVector &F, cudaCVector &f)
{
	int NX = (int)F.get_N();
	int BATCH = 1;

	cufftHandle plan;
	if (cufftPlan1d(&plan, NX, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleComplex*)f.get_Array(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.get_N() / (double)BLOCK_SIZE));
	kernelNorm <<<grid, block>>> (NX, f.get_Array());
	cudaDeviceSynchronize();
}
void cuifft(cudaCVector &F, cudaRVector &f)
{
	int NX = (int)F.get_N();
	int BATCH = 1;

	cufftHandle plan;
	if (cufftPlan1d(&plan, NX, CUFFT_Z2D, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecZ2D(plan, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleReal*)f.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)NX / (double)BLOCK_SIZE));
	kernelNorm <<<grid, block>>> (NX, f.get_Array());
	cudaDeviceSynchronize();
}
void cuifft(cudaCVector3 &F, cudaCVector3 &f)
{
	int NX = (int)F.get_N1();
	int NY = (int)F.get_N2();
	int NZ = (int)F.get_N3();

	cufftHandle plan;
	if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_Z2Z) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleComplex*)f.get_Array(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)NX*NY*NZ / (double)BLOCK_SIZE));
	kernelNorm<<<grid, block>>>(NX*NY*NZ, f.get_Array());
	cudaDeviceSynchronize();

}
void cuifft(cudaCVector3 &F, cudaRVector3 &f)
{
	int NX = (int)F.get_N1();
	int NY = (int)F.get_N2();
	int NZ = (int)F.get_N3();

	cufftHandle plan;
	if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_Z2D) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	if (cufftExecZ2D(plan, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleReal*)f.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cufftDestroy(plan);
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)NX*NY*NZ / (double)BLOCK_SIZE));
	kernelNorm<<<grid, block>>>(NX*NY*NZ, f.get_Array());
	cudaDeviceSynchronize();
}

void cuFFT::forward(cudaCVector &f, cudaCVector &F)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)f.get_Array(), (cufftDoubleComplex*)F.get_Array(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
}
void cuFFT::forward(cudaRVector &f, cudaCVector &F)
{
	if (cufftExecD2Z(planD2Z, (cufftDoubleReal*)f.get_Array(), (cufftDoubleComplex*)F.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
}
void cuFFT::forward(cudaCVector3 &f, cudaCVector3 &F)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)f.get_Array(), (cufftDoubleComplex*)F.get_Array(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecZ2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
}
void cuFFT::forward(cudaRVector3 &f, cudaCVector3 &F)
{
	if (cufftExecD2Z(planD2Z, (cufftDoubleReal*)f.get_Array(), (cufftDoubleComplex*)F.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecD2Z Forward failed");
		return;
	}
	cudaDeviceSynchronize();
}

void cuFFT::inverce(cudaCVector &F, cudaCVector &f)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleComplex*)f.get_Array(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.get_N() / (double)BLOCK_SIZE));
	kernelNorm<<<grid, block>>>((int)f.get_N(), f.get_Array());
	cudaDeviceSynchronize();
}
void cuFFT::inverce(cudaCVector &F, cudaRVector &f)
{
	if (cufftExecZ2D(planZ2D, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleReal*)f.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.get_N() / (double)BLOCK_SIZE));
	kernelNorm<<<grid, block>>>((int)f.get_N(), f.get_Array());
	cudaDeviceSynchronize();
}
void cuFFT::inverce(cudaCVector3 &F, cudaCVector3 &f)
{
	if (cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleComplex*)f.get_Array(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.size() / (double)BLOCK_SIZE));
	kernelNorm<<<grid, block>>> ((int)f.size(), f.get_Array());
	cudaDeviceSynchronize();
}
void cuFFT::inverce(cudaCVector3 &F, cudaRVector3 &f)
{
	if (cufftExecZ2D(planZ2D, (cufftDoubleComplex*)F.get_Array(), (cufftDoubleReal*)f.get_Array()) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: 3D ExecZ2Z Inverce failed");
		return;
	}
	cudaDeviceSynchronize();

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)f.size() / (double)BLOCK_SIZE));
	kernelNorm<<<grid, block>>>((int)f.size(), f.get_Array());
	cudaDeviceSynchronize();
}

cuFFT::cuFFT(const int _dim, const int *_n, const int _BATCH) : dim(_dim), BATCH(_BATCH)
{
	n = new int[dim];
	for (size_t i = 0; i < dim; i++)
		n[i] = _n[i];

	int NX, NY, NZ;

	switch (dim)
	{
	case 1:
		NX = n[0];

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
void cuFFT::reset(const int _dim, const int *_n, const int _BATCH)
{
	dim = _dim;
	BATCH = _BATCH;

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