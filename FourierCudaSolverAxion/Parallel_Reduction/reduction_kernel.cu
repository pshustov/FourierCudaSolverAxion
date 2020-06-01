#include "reduction.h"

namespace cg = cooperative_groups;

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

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void kernekReduce(T* g_idata, T* g_odata, unsigned int n, T (*fun)(T, T))
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


template <typename T>
void reduce(int size, int threads, int blocks, T(*fun)(T, T), T* d_idata, T* d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
				kernekReduce<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
				kernekReduce<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
				kernekReduce<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
				kernekReduce<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
				kernekReduce<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
				kernekReduce<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
				kernekReduce<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
				kernekReduce<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
				kernekReduce<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  1:
				kernekReduce<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
				kernekReduce<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
				kernekReduce<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
				kernekReduce<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
				kernekReduce<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
				kernekReduce<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
				kernekReduce<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
				kernekReduce<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
				kernekReduce<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
				kernekReduce<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  1:
				kernekReduce<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
}



