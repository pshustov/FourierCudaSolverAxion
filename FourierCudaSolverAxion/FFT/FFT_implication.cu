#include "stdafx.h"

#ifdef _WIN64
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
#endif

#ifdef linux
__device__ void callbackForwardNormZ(void* dataOut, size_t offset, cufftDoubleComplex element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	((complex*)dataOut)[offset] *= (dataLN[0] / dataLN[1]);
}
__device__ void callbackForwardNormD(void* dataOut, size_t offset, cufftDoubleReal element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	((double*)dataOut)[offset] *= (dataLN[0] / dataLN[1]);
}
__device__ void callbackInverseNormZ(void* dataOut, size_t offset, cufftDoubleComplex element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	((complex*)dataOut)[offset] /= dataLN[0];
}
__device__ void callbackInverseNormD(void* dataOut, size_t offset, cufftDoubleReal element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	((double*)dataOut)[offset] /= dataLN[0];
}

__device__ cufftCallbackStoreZ d_callbackForwardNormZ = callbackForwardNormZ;
__device__ cufftCallbackStoreD d_callbackForwardNormD = callbackForwardNormD;
__device__ cufftCallbackStoreZ d_callbackInverseNormZ = callbackInverseNormZ;
__device__ cufftCallbackStoreD d_callbackInverseNormD = callbackInverseNormD;
#endif

void cuFFT::forward(cudaCVector3& f, cudaCVector3& F, bool isNormed)
{
	checkCudaErrors(cufftExecZ2Z(planZ2ZF, (cufftDoubleComplex*)f.getArray(), (cufftDoubleComplex*)F.getArray(), CUFFT_FORWARD));
	checkCudaErrors(cudaStreamSynchronize(stream));

#ifdef _WIN64
#endif
}
void cuFFT::forward(cudaRVector3& f, cudaCVector3& F, bool isNormed)
{
	checkCudaErrors(cufftExecD2Z(planD2Z, (cufftDoubleReal*)f.getArray(), (cufftDoubleComplex*)F.getArray()));
	checkCudaErrors(cudaStreamSynchronize(stream));
}
void cuFFT::inverce(cudaCVector3 &F, cudaCVector3 &f, bool isNormed)
{
	checkCudaErrors(cufftExecZ2Z(planZ2ZI, (cufftDoubleComplex*)F.getArray(), (cufftDoubleComplex*)f.getArray(), CUFFT_INVERSE));
	checkCudaErrors(cudaStreamSynchronize(stream));
}
void cuFFT::inverce(cudaCVector3 &F, cudaRVector3 &f, bool isNormed)
{
	checkCudaErrors(cufftExecZ2D(planZ2D, (cufftDoubleComplex*)F.getArray(), (cufftDoubleReal*)f.getArray()));
	checkCudaErrors(cudaStreamSynchronize(stream));
}


cuFFT::cuFFT(cudaStream_t _stream) : stream(_stream)
{
	dim = 1;
	n = new int[dim];
	n[0] = 1024;
	L = 10;
	N = 1024;

	BATCH = 1;

	checkCudaErrors(cufftCreate(&planZ2ZF));
	checkCudaErrors(cufftCreate(&planZ2ZI));
	checkCudaErrors(cufftCreate(&planD2Z));
	checkCudaErrors(cufftCreate(&planZ2D));

	setStream(stream);

#ifdef linux
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormZ, d_callbackForwardNormZ, sizeof(h_callbackForwardNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormD, d_callbackForwardNormD, sizeof(h_callbackForwardNormD)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormZ, d_callbackInverseNormZ, sizeof(h_callbackInverseNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormD, d_callbackInverseNormD, sizeof(h_callbackInverseNormD)));
#endif

	callbackData = new double[2];

	callbackData[0] = L;
	callbackData[1] = N;
}

cuFFT::cuFFT(const int _dim, const int *_n, const int _BATCH, cudaStream_t _stream) : dim(_dim), BATCH(_BATCH), stream(_stream)
{
	throw;
	n = new int[dim];
	for (size_t i = 0; i < dim; i++)
		n[i] = _n[i];

	int NX, NY, NZ;
	L = 1;

	setStream(stream);

	switch (dim)
	{
	case 1:
		/*NX = n[0];
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
		break;*/
		throw;
	case 3:
		NX = n[0];
		NY = n[1];
		NZ = n[2];
		N = NX * NY * NZ;

		if (cufftPlan3d(&planZ2ZF, NX, NY, NZ, CUFFT_Z2Z) != CUFFT_SUCCESS) {
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
	checkCudaErrors(cufftDestroy(planZ2ZF));
	checkCudaErrors(cufftDestroy(planZ2ZI));
	checkCudaErrors(cufftDestroy(planD2Z));
	checkCudaErrors(cufftDestroy(planZ2D));
	delete[] n;
	delete[] callbackData;
}
void cuFFT::reset(const int _dim, const int *_n, double _L, const int _BATCH, cudaStream_t _stream)
{
	dim = _dim;
	delete[] n;
	n = new int[dim];
	N = 1;
	for (size_t i = 0; i < dim; i++) {
		n[i] = _n[i];
		N *= n[i];
	}
	
	BATCH = _BATCH;
	L = _L;

	callbackData[0] = L;
	callbackData[1] = N;

	checkCudaErrors(cufftDestroy(planZ2ZF));
	checkCudaErrors(cufftDestroy(planZ2ZI));
	checkCudaErrors(cufftDestroy(planD2Z));
	checkCudaErrors(cufftDestroy(planZ2D));

	switch (dim)
	{
	case 1:
		/*NX = n[0];
		N = NX;

		if (cufftPlan1d(&planZ2Z, NX, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			throw;
		}
		if (cufftPlan1d(&planD2Z, NX, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			throw;
		}
		if (cufftPlan1d(&planZ2D, NX, CUFFT_Z2D, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			throw;
		}
		break;*/
		throw;

	case 3:
		size_t workSize;
		
		checkCudaErrors(cufftCreate(&planZ2ZF));
		checkCudaErrors(cufftMakePlan3d(planZ2ZF, n[0], n[1], n[2], CUFFT_Z2Z, &workSize));

		checkCudaErrors(cufftCreate(&planZ2ZI));
		checkCudaErrors(cufftMakePlan3d(planZ2ZI, n[0], n[1], n[2], CUFFT_Z2Z, &workSize));

		checkCudaErrors(cufftCreate(&planD2Z));
		checkCudaErrors(cufftMakePlan3d(planD2Z, n[0], n[1], n[2], CUFFT_D2Z, &workSize));

		checkCudaErrors(cufftCreate(&planZ2D));
		checkCudaErrors(cufftMakePlan3d(planZ2D, n[0], n[1], n[2], CUFFT_Z2D, &workSize));

#ifdef linux
		checkCudaErrors(cufftXtSetCallback(planZ2ZF, (void**)&h_callbackForwardNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)callbackData));
		checkCudaErrors(cufftXtSetCallback(planZ2ZI, (void**)&h_callbackInverseNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)callbackData));
		checkCudaErrors(cufftXtSetCallback(planD2Z, (void**)&h_callbackForwardNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)callbackData));
		checkCudaErrors(cufftXtSetCallback(planZ2D, (void**)&h_callbackInverseNormD, CUFFT_CB_ST_REAL_DOUBLE, (void**)callbackData));
#endif

		//if (cufftPlan3d(&planZ2Z, NX, NY, NZ, CUFFT_Z2Z) != CUFFT_SUCCESS) {
		//	fprintf(stderr, "CUFFT error: Plan creation failed");
		//	throw;
		//}
		//if (cufftPlan3d(&planD2Z, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS) {
		//	fprintf(stderr, "CUFFT error: Plan creation failed");
		//	throw;
		//}
		//if (cufftPlan3d(&planZ2D, NX, NY, NZ, CUFFT_Z2D) != CUFFT_SUCCESS) {
		//	fprintf(stderr, "CUFFT error: Plan creation failed");
		//	throw;
		//}

		break;

	default:
		throw;
	}

	stream = _stream;
	setStream(stream);
	std::cout << "55" << std::endl;
}