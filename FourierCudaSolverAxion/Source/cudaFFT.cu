#include <device_launch_parameters.h>

#include "cudaVector.h"
#include "cudaFFT.h"

#include <helper_cuda.h>

#define FFT_BLOCK_SIZE 128

#ifdef __linux__

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

#else

__global__ void kernelForwardNorm(const size_t size, const size_t N, const real L, real* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] * L / N;
	}
}
__global__ void kernelForwardNorm(const size_t size, const size_t N, const real L, complex* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] * L / N;
	}
}
__global__ void kernelInverseNorm(const size_t size, const size_t N, const real L, real* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] / L;
	}
}
__global__ void kernelInverseNorm(const size_t size, const size_t N, const real L, complex* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] / L;
	}
}

#endif // __linux__

void cuFFT::forward(cudaCVector3& f, cudaCVector3& F)
{
	checkCudaErrors(cufftXtExec(planC2CF, f.getArray(), F.getArray(), CUFFT_FORWARD));
	
#ifndef __linux__
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(F.size()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelForwardNorm <<< grid, block, 0, stream >>> (F.size(), N, L, F.getArray());
#endif // !__linux__
}
void cuFFT::forward(cudaRVector3& f, cudaCVector3& F)
{
	checkCudaErrors(cufftXtExec(planR2C, f.getArray(), F.getArray(), CUFFT_FORWARD));

#ifndef __linux__
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(F.size()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelForwardNorm << < grid, block, 0, stream >> > (F.size(), N, L, F.getArray());
#endif // !__linux__
}
void cuFFT::inverce(cudaCVector3& F, cudaCVector3& f)
{
	checkCudaErrors(cufftXtExec(planC2CI, F.getArray(), f.getArray(), CUFFT_INVERSE));
	
#ifndef __linux__
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(f.size()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelInverseNorm << < grid, block, 0, stream >> > (f.size(), N, L, f.getArray());
#endif // !__linux__
}
void cuFFT::inverce(cudaCVector3& F, cudaRVector3& f)
{
	checkCudaErrors(cufftXtExec(planC2R, F.getArray(), f.getArray(), CUFFT_INVERSE));
	//checkCudaErrors(cufftExecC2R(planC2R, (cufftComplex*)F.getArray(), (cufftReal*)f.getArray()));
	//checkCudaErrors(cufftExecC2R(planC2R, (cufftComplex*)F.getArray(), (cufftReal*)f.getArray()));


#ifndef __linux__
	dim3 block(FFT_BLOCK_SIZE);
	dim3 grid((static_cast<unsigned int>(f.size()) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	kernelInverseNorm << < grid, block, 0, stream >> > (f.size(), N, L, f.getArray());
#endif // !__linux__
}

void cuFFT::setStream(cudaStream_t stream) 
{
	checkCudaErrors(cufftSetStream(planC2CF, stream));
	checkCudaErrors(cufftSetStream(planC2CI, stream));
	checkCudaErrors(cufftSetStream(planR2C, stream));
	checkCudaErrors(cufftSetStream(planC2R, stream));
}

cuFFT::cuFFT(cudaStream_t _stream) : stream(_stream)
{
	dim = 1;
	n = new int[dim];
	n[0] = 1024;
	L = 10;
	N = 1024;

	BATCH = 1;

	checkCudaErrors(cufftCreate(&planC2CF));
	checkCudaErrors(cufftCreate(&planC2CI));
	checkCudaErrors(cufftCreate(&planR2C));
	checkCudaErrors(cufftCreate(&planC2R));

	setStream(stream);

#ifdef __linux__
	std::cout << "LINUX detected" << std::endl;
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormZ, d_callbackForwardNormZ, sizeof(h_callbackForwardNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormD, d_callbackForwardNormD, sizeof(h_callbackForwardNormD)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormZ, d_callbackInverseNormZ, sizeof(h_callbackInverseNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormD, d_callbackInverseNormD, sizeof(h_callbackInverseNormD)));
#endif

	checkCudaErrors(cudaMallocManaged(&callbackData, 2 * sizeof(double)));
	callbackData[0] = L;
	callbackData[1] = N;
}

cuFFT::cuFFT(const int _dim, const int *_n, real _L, const int _BATCH, cudaStream_t _stream) : dim(_dim), BATCH(_BATCH), stream(_stream)
{
	dim = 1;
	n = new int[dim];
	n[0] = 1024;
	L = 10;
	N = 1024;

	BATCH = 1;

	checkCudaErrors(cufftCreate(&planC2CF));
	checkCudaErrors(cufftCreate(&planC2CI));
	checkCudaErrors(cufftCreate(&planR2C));
	checkCudaErrors(cufftCreate(&planC2R));

	setStream(stream);

#ifdef __linux__
	std::cout << "LINUX detected" << std::endl;
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormZ, d_callbackForwardNormZ, sizeof(h_callbackForwardNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackForwardNormD, d_callbackForwardNormD, sizeof(h_callbackForwardNormD)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormZ, d_callbackInverseNormZ, sizeof(h_callbackInverseNormZ)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_callbackInverseNormD, d_callbackInverseNormD, sizeof(h_callbackInverseNormD)));
#endif

	checkCudaErrors(cudaMallocManaged(&callbackData, 2 * sizeof(double)));
	callbackData[0] = L;
	callbackData[1] = N;

	reset(_dim, _n, _L, _BATCH, _stream);
}
cuFFT::~cuFFT()
{
	checkCudaErrors(cufftDestroy(planC2CF));
	checkCudaErrors(cufftDestroy(planC2CI));
	checkCudaErrors(cufftDestroy(planR2C));
	checkCudaErrors(cufftDestroy(planC2R));
	checkCudaErrors(cudaFree(callbackData));
	delete[] n;
}
void cuFFT::reset(const int _dim, const int *_n, real _L, const int _BATCH, cudaStream_t _stream)
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

	checkCudaErrors(cufftDestroy(planC2CF));
	checkCudaErrors(cufftDestroy(planC2CI));
	checkCudaErrors(cufftDestroy(planR2C));
	checkCudaErrors(cufftDestroy(planC2R));

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
		throw;

	case 3:
		size_t workSize;
		
#ifdef _WIN64
		if (typeid(real)==typeid(double)) {
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
				throw;
			}
		}

#endif // _WIN64


#ifdef __linux__
		checkCudaErrors(cufftCreate(&planC2CF));
		checkCudaErrors(cufftMakePlan3d(planC2CF, n[0], n[1], n[2], CUFFT_Z2Z, &workSize));

		checkCudaErrors(cufftCreate(&planC2CI));
		checkCudaErrors(cufftMakePlan3d(planC2CI, n[0], n[1], n[2], CUFFT_Z2Z, &workSize));

		checkCudaErrors(cufftCreate(&planR2C));
		checkCudaErrors(cufftMakePlan3d(planR2C, n[0], n[1], n[2], CUFFT_D2Z, &workSize));

		checkCudaErrors(cufftCreate(&planC2R));
		checkCudaErrors(cufftMakePlan3d(planC2R, n[0], n[1], n[2], CUFFT_Z2D, &workSize));

		std::cout << "LINUX detected" << std::endl;
		checkCudaErrors(cufftXtSetCallback(planC2CF, (void**)&h_callbackForwardNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)&callbackData));
		checkCudaErrors(cufftXtSetCallback(planC2CI, (void**)&h_callbackInverseNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)&callbackData));
		checkCudaErrors(cufftXtSetCallback(planR2C, (void**)&h_callbackForwardNormZ, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)&callbackData));
		checkCudaErrors(cufftXtSetCallback(planC2R, (void**)&h_callbackInverseNormD, CUFFT_CB_ST_REAL_DOUBLE, (void**)&callbackData));
#endif	// __linux__

		break;

	default:
		throw;
	}

	stream = _stream;
	setStream(stream);
}