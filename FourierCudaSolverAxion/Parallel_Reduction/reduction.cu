#include "stdafx.h"

namespace cg = cooperative_groups;


bool isPow2(unsigned int x)
{
	return ((x & (x - 1)) == 0);
}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

void getNumBlocksAndThreads(int n, int maxThreads, int& blocks, int& threads)
{
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);

	if ((float)threads * blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
	{
		printf("n is too large, please choose a smaller number!\n");
	}

	if (blocks > prop.maxGridSize[0])
	{
		printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
			blocks, prop.maxGridSize[0], threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<typename T>
struct SharedMemory
{
	__device__ inline operator T* ()
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}

	__device__ inline operator const T* () const
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double* ()
	{
		extern __shared__ double __smem_d[];
		return (double*)__smem_d;
	}

	__device__ inline operator const double* () const
	{
		extern __shared__ double __smem_d[];
		return (double*)__smem_d;
	}
};

template <typename T, unsigned int blockSize, bool nIsPow2, typename F>
__global__ void kernelReduce(T* g_idata, T* g_odata, unsigned int n, F fun)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T result = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		result = fun(result, g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			result = fun(result, g_idata[i + blockSize]);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = result;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = result = fun(result, sdata[tid + 256]);
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = result = fun(result, sdata[tid + 128]);
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = result = fun(result, sdata[tid + 64]);
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) result = fun(result, sdata[tid + 32]);
		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
		{
			result = fun(result, tile32.shfl_down(result, offset));
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) g_odata[blockIdx.x] = result;
}

template <typename T, unsigned int blockSize, bool nIsPow2, typename F>
__global__ void kernelReduce2(T* g_idata, T* g_odata, unsigned int n, F fun)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T result = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		result = fun(result, g_idata[i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n) {
			result = fun(result, g_idata[i + blockSize]);
		}

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = result;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = result = fun(result, sdata[tid + 256]);
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = result = fun(result, sdata[tid + 128]);
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = result = fun(result, sdata[tid + 64]);
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) result = fun(result, sdata[tid + 32]);
		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 1; offset /= 2)
		{
			result = fun(result, tile32.shfl_down(result, offset));
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) { 
		g_odata[2*blockIdx.x] = result; 
	}
	if (cta.thread_rank() == 1) {
		g_odata[2*blockIdx.x + 1] = result; 
	}
}



template <typename T, typename F>
void reduce(int size, int threads, int blocks, F fun, T* d_idata, T* d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
				kernelReduce<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
				kernelReduce<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
				kernelReduce<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
				kernelReduce<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
				kernelReduce<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
				kernelReduce<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
				kernelReduce<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
				kernelReduce<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
				kernelReduce<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  1:
				kernelReduce<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
				kernelReduce<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
				kernelReduce<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
				kernelReduce<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
				kernelReduce<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
				kernelReduce<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
				kernelReduce<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
				kernelReduce<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
				kernelReduce<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
				kernelReduce<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  1:
				kernelReduce<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
}

template <typename T, typename F>
void reduce2(int size, int threads, int blocks, F fun, T* d_idata, T* d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
				kernelReduce2<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
				kernelReduce2<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
				kernelReduce2<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
				kernelReduce2<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
				kernelReduce2<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
				kernelReduce2<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
				kernelReduce2<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
				kernelReduce2<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
				kernelReduce2<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
				kernelReduce2<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
				kernelReduce2<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
				kernelReduce2<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
				kernelReduce2<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
				kernelReduce2<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
				kernelReduce2<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
				kernelReduce2<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
				kernelReduce2<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
				kernelReduce2<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
}



template <typename T>
T reductionSum(int size, T* inData)
{
	int cpuFinalThreshold = 256;
	int maxThreads = 256;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);

	T* inData_dev = NULL;
	T* outData_dev = NULL;

	cudaMalloc((void**)&inData_dev, blocks * sizeof(T));
	cudaMalloc((void**)&outData_dev, blocks * sizeof(T));

	auto fun = [] __host__ __device__(T A, T B) { return A + B; };

	reduce(size, threads, blocks, fun, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(T), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce(s, threads, blocks, fun, inData_dev, outData_dev);

		s = blocks;
	}

	T* outData_host;
	outData_host = (T*)malloc(s * sizeof(T));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(T), cudaMemcpyDeviceToHost);

	T result = 0;
	for (size_t i = 0; i < s; i++)
	{
		result = fun(result, outData_host[i]);
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}
template int reductionSum<int>(int size, int* inData);
template float reductionSum<float>(int size, float* inData);
template double reductionSum<double>(int size, double* inData);

template <>
complex reductionSum<complex>(int size, complex* inData)
{
	int cpuFinalThreshold = 1;
	int maxThreads = 16;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);

	complex* inData_dev = NULL;
	complex* outData_dev = NULL;

	cudaMalloc((void**)&inData_dev, blocks * sizeof(complex));
	cudaMalloc((void**)&outData_dev, blocks * sizeof(complex));

	auto fun = [] __host__ (complex A, complex B) { return A + B; };
	auto funDub = [] __device__(double A, double B) { return A + B; };

	reduce2(2*size, 2*threads, blocks, funDub, (double*)inData, (double*)outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(complex), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce2(2*s, 2*threads, blocks, funDub, (double*)inData_dev, (double*)outData_dev);

		s = blocks;
	}

	complex* outData_host;
	outData_host = (complex*)malloc(s * sizeof(complex));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(complex), cudaMemcpyDeviceToHost);

	complex result = 0;
	for (size_t i = 0; i < s; i++)
	{
		result = fun(result, outData_host[i]);
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}