#include <device_launch_parameters.h>

#include "cudaVector.h"
#include "cudaFFT.h"

#include <helper_cuda.h>

// #define CALBACKS
#define FFT_BLOCK_SIZE 128

#ifdef CALBACKS

__device__ void callbackForwardNormZ(void* dataOut, size_t offset, cufftDoubleComplex element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	complex el = *(complex*)&element;
	((complex*)dataOut)[offset] = el * (dataLN[0] / dataLN[1]);
}
__device__ void callbackForwardNormD(void* dataOut, size_t offset, cufftDoubleReal element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	((double*)dataOut)[offset] = element * (dataLN[0] / dataLN[1]);
}
__device__ void callbackInverseNormZ(void* dataOut, size_t offset, cufftDoubleComplex element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	complex el = *(complex*)&element;
	((complex*)dataOut)[offset] = el / dataLN[0];
}
__device__ void callbackInverseNormD(void* dataOut, size_t offset, cufftDoubleReal element, void* callerInfo, void* sharedPointer)
{
	double* dataLN = (double*)callerInfo;
	((double*)dataOut)[offset] = element / dataLN[0];
}

__device__ cufftCallbackStoreZ d_callbackForwardNormZ = callbackForwardNormZ;
__device__ cufftCallbackStoreD d_callbackForwardNormD = callbackForwardNormD;
__device__ cufftCallbackStoreZ d_callbackInverseNormZ = callbackInverseNormZ;
__device__ cufftCallbackStoreD d_callbackInverseNormD = callbackInverseNormD;


__device__ void callbackForwardNormC(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPointer)
{
	float* dataVandN = (float*)callerInfo;
	complex el = *(complex*)&element;
	((complex*)dataOut)[offset] = el * (dataVandN[0] / dataVandN[1]);
}
__device__ void callbackForwardNormR(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPointer)
{
	float* dataVandN = (float*)callerInfo;
	((float*)dataOut)[offset] = element * (dataVandN[0] / dataVandN[1]);
}
__device__ void callbackInverseNormC(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPointer)
{
	float* dataVandN = (float*)callerInfo;
	complex el = *(complex*)&element;
	((complex*)dataOut)[offset] = el / dataVandN[0];
}
__device__ void callbackInverseNormR(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPointer)
{
	float* dataVandN = (float*)callerInfo;
	((float*)dataOut)[offset] = element / dataVandN[0];
}

__device__ cufftCallbackStoreC d_callbackForwardNormC = callbackForwardNormC;
__device__ cufftCallbackStoreR d_callbackForwardNormR = callbackForwardNormR;
__device__ cufftCallbackStoreC d_callbackInverseNormC = callbackInverseNormC;
__device__ cufftCallbackStoreR d_callbackInverseNormR = callbackInverseNormR;

#else

__global__ void kernelForwardNorm(const size_t size, const size_t N, const real vol, real* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] *= (vol / N);
	}
}
__global__ void kernelForwardNorm(const size_t size, const size_t N, const real vol, complex* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] *= (vol / N);
	}
}
__global__ void kernelInverseNorm(const size_t size, const size_t N, const real vol, real* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] /= vol;
	}
}
__global__ void kernelInverseNorm(const size_t size, const size_t N, const real vol, complex* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] /= vol;
	}
}

#endif // CALBACKS

void cuFFT::forward(cudaCVector3& f, cudaCVector3& F)
{
	checkCudaErrors(cufftXtExec(planC2CF, f.getArray(), F.getArray(), CUFFT_FORWARD));
	
#ifndef CALBACKS
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(F.getSize()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelForwardNorm <<< grid, block, 0, stream >>> (F.getSize(), N, volume, F.getArray());
#endif // !CALBACKS
}
void cuFFT::forward(cudaRVector3& f, cudaCVector3& F)
{
	checkCudaErrors(cufftXtExec(planR2C, f.getArray(), F.getArray(), CUFFT_FORWARD));

#ifndef CALBACKS
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(F.getSize()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelForwardNorm << < grid, block, 0, stream >> > (F.getSize(), N, volume, F.getArray());
#endif // !CALBACKS
}
void cuFFT::inverce(cudaCVector3& F, cudaCVector3& f)
{
	checkCudaErrors(cufftXtExec(planC2CI, F.getArray(), f.getArray(), CUFFT_INVERSE));
	
#ifndef CALBACKS
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(f.getSize()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelInverseNorm << < grid, block, 0, stream >> > (f.getSize(), N, volume, f.getArray());
#endif // !CALBACKS
}
void cuFFT::inverce(cudaCVector3& F, cudaRVector3& f)
{
	checkCudaErrors(cufftXtExec(planC2R, F.getArray(), f.getArray(), CUFFT_INVERSE));

#ifndef CALBACKS
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(f.getSize()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelInverseNorm <<< grid, block, 0, stream >>> (f.getSize(), N, volume, f.getArray());
#endif // !CALBACKS
}

void cuFFT::setStream(cudaStream_t stream) 
{
	checkCudaErrors(cufftSetStream(planC2CF, stream));
	checkCudaErrors(cufftSetStream(planC2CI, stream));
	checkCudaErrors(cufftSetStream(planR2C, stream));
	checkCudaErrors(cufftSetStream(planC2R, stream));
}

cuFFT::cuFFT()
{
	isInitialized = false;
#ifdef CALBACKS
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormC, d_callbackForwardNormC, sizeof(h_callbackForwardNormC)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormR, d_callbackForwardNormR, sizeof(h_callbackForwardNormR)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormC, d_callbackInverseNormC, sizeof(h_callbackInverseNormC)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormR, d_callbackInverseNormR, sizeof(h_callbackInverseNormR)));

	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormZ, d_callbackForwardNormZ, sizeof(h_callbackForwardNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormD, d_callbackForwardNormD, sizeof(h_callbackForwardNormD)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormZ, d_callbackInverseNormZ, sizeof(h_callbackInverseNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormD, d_callbackInverseNormD, sizeof(h_callbackInverseNormD)));
#endif // CALBACKS
}
cuFFT::cuFFT(const int _dim, const int *_n, real _volume, const int _BATCH, cudaStream_t _stream) : dim(_dim), volume(_volume), BATCH(_BATCH), stream(_stream)
{
#ifdef CALBACKS
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormC, d_callbackForwardNormC, sizeof(h_callbackForwardNormC)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormR, d_callbackForwardNormR, sizeof(h_callbackForwardNormR)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormC, d_callbackInverseNormC, sizeof(h_callbackInverseNormC)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormR, d_callbackInverseNormR, sizeof(h_callbackInverseNormR)));

	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormZ, d_callbackForwardNormZ, sizeof(h_callbackForwardNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormD, d_callbackForwardNormD, sizeof(h_callbackForwardNormD)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormZ, d_callbackInverseNormZ, sizeof(h_callbackInverseNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormD, d_callbackInverseNormD, sizeof(h_callbackInverseNormD)));
#endif // CALBACKS
	reset(_dim, _n, _volume, _BATCH, _stream);
}
cuFFT::~cuFFT()
{
	clear();
}
void cuFFT::reset(const int _dim, const int *_n, real _volume, const int _BATCH, cudaStream_t _stream)
{
	clear();

	dim = _dim;
	n = new int[dim];
	N = 1;
	for (size_t i = 0; i < dim; i++) {
		n[i] = _n[i];
		N *= n[i];
	}
	
	BATCH = _BATCH;
	volume = _volume;

	switch (dim)
	{
	case 1:
		/*NX = n[0];
		N = NX;

		if (cufftPlan1d(&planZ2Z, NX, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			throw;
		}
		if (cufftPlan1d(&planR2C, NX, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			throw;
		}
		if (cufftPlan1d(&planC2R, NX, CUFFT_Z2D, BATCH) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: Plan creation failed");
			throw;
		}
		break;*/
		throw -1;
	case 3:		
#ifdef CALBACKS
		std::cout << "Callbacks enabled" << std::endl;

		size_t workSize;

		checkCudaErrors(cufftCreate(&planC2CF));
		checkCudaErrors(cufftCreate(&planC2CI));
		checkCudaErrors(cufftCreate(&planR2C));
		checkCudaErrors(cufftCreate(&planC2R));

		checkCudaErrors(cudaMallocManaged(&callbackData, 2 * sizeof(real)));
		callbackData[0] = volume;
		callbackData[1] = (real)N;

		if (typeid(real) == typeid(double)) {
			checkCudaErrors(cufftMakePlan3d(planC2CF, n[0], n[1], n[2], CUFFT_Z2Z, &workSize));
			checkCudaErrors(cufftMakePlan3d(planC2CI, n[0], n[1], n[2], CUFFT_Z2Z, &workSize));
			checkCudaErrors(cufftMakePlan3d(planR2C, n[0], n[1], n[2], CUFFT_D2Z, &workSize));
			checkCudaErrors(cufftMakePlan3d(planC2R, n[0], n[1], n[2], CUFFT_Z2D, &workSize));
			checkCudaErrors(cufftXtSetCallback(planC2CF, (void**)&h_callbackForwardNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)&callbackData));
			checkCudaErrors(cufftXtSetCallback(planC2CI, (void**)&h_callbackInverseNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)&callbackData));
			checkCudaErrors(cufftXtSetCallback(planR2C, (void**)&h_callbackForwardNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)&callbackData));
			checkCudaErrors(cufftXtSetCallback(planC2R, (void**)&h_callbackInverseNormD, CUFFT_CB_ST_REAL_DOUBLE, (void**)&callbackData));
		}
		else {
			if (typeid(real) == typeid(float)) {
				checkCudaErrors(cufftMakePlan3d(planC2CF, n[0], n[1], n[2], CUFFT_C2C, &workSize));
				checkCudaErrors(cufftMakePlan3d(planC2CI, n[0], n[1], n[2], CUFFT_C2C, &workSize));
				checkCudaErrors(cufftMakePlan3d(planR2C, n[0], n[1], n[2], CUFFT_R2C, &workSize));
				checkCudaErrors(cufftMakePlan3d(planC2R, n[0], n[1], n[2], CUFFT_C2R, &workSize));
				checkCudaErrors(cufftXtSetCallback(planC2CF, (void**)&h_callbackForwardNormC, CUFFT_CB_ST_COMPLEX, (void**)&callbackData));
				checkCudaErrors(cufftXtSetCallback(planC2CI, (void**)&h_callbackInverseNormC, CUFFT_CB_ST_COMPLEX, (void**)&callbackData));
				checkCudaErrors(cufftXtSetCallback(planR2C, (void**)&h_callbackForwardNormC, CUFFT_CB_ST_COMPLEX, (void**)&callbackData));
				checkCudaErrors(cufftXtSetCallback(planC2R, (void**)&h_callbackInverseNormR, CUFFT_CB_ST_REAL, (void**)&callbackData));
			}
			else {
				throw - 1;
			}
		}
#else
		std::cout << "Callbacks disabled" << std::endl;

		if (typeid(real) == typeid(double)) {
			checkCudaErrors(cufftPlan3d(&planC2CF, n[0], n[1], n[2], CUFFT_Z2Z));
			checkCudaErrors(cufftPlan3d(&planC2CI, n[0], n[1], n[2], CUFFT_Z2Z));
			checkCudaErrors(cufftPlan3d(&planR2C, n[0], n[1], n[2], CUFFT_D2Z));
			checkCudaErrors(cufftPlan3d(&planC2R, n[0], n[1], n[2], CUFFT_Z2D));
		}
		else {
			if (typeid(real) == typeid(float)) {
				checkCudaErrors(cufftPlan3d(&planC2CF, n[0], n[1], n[2], CUFFT_C2C));
				checkCudaErrors(cufftPlan3d(&planC2CI, n[0], n[1], n[2], CUFFT_C2C));
				checkCudaErrors(cufftPlan3d(&planR2C, n[0], n[1], n[2], CUFFT_R2C));
				checkCudaErrors(cufftPlan3d(&planC2R, n[0], n[1], n[2], CUFFT_C2R));
			}
			else {
				throw - 1;
			}
		}
#endif // CALBACKS
		break;
	default:
		throw -1;
	}

	stream = _stream;
	setStream(stream);

	isInitialized = true;
}

void cuFFT::clear()
{
	if (isInitialized) {
		delete[] n;
		checkCudaErrors(cufftDestroy(planC2CF));
		checkCudaErrors(cufftDestroy(planC2CI));
		checkCudaErrors(cufftDestroy(planR2C));
		checkCudaErrors(cufftDestroy(planC2R));
#ifdef CALBACKS
		checkCudaErrors(cudaFree(callbackData));
#endif // CALBACKS
		isInitialized = false;
	}
}