#include "reduction.h"

namespace cg = cooperative_groups;

__device__ double getMax(double x, double y) {
	return (x > y) ? x : y;
}

__device__ double getSum(double x, double y) {
	return x + y;
}

__global__ void reduceKernelMax2(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i] : 0;

	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = getMax(sdata[tid], sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduceKernelMax3(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getMax(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = result = getMax(result, sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = result;
}
template <unsigned int blockSize>
__global__ void reduceKernelMax4(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getMax(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = result = getMax(result, sdata[tid + s]);
		}
		cg::sync(cta);
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		if (blockSize >= 64) result = getMax(result, sdata[tid + 32]);

		for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
		{
			result = getMax(result, tile32.shfl_down(result, offset));
		}
	}


	if (tid == 0) g_odata[blockIdx.x] = result;
}
template <unsigned int blockSize>
__global__ void reduceKernelMax5(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getMax(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = result = getMax(result, sdata[tid + 256]);
	}
	cg::sync(cta);


	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = result = getMax(result, sdata[tid + 128]);
	}
	cg::sync(cta);


	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = result = getMax(result, sdata[tid + 64]);
	}
	cg::sync(cta);


	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		if (blockSize >= 64) result = getMax(result, sdata[tid + 32]);

		for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
		{
			result = getMax(result, tile32.shfl_down(result, offset));
		}
	}


	if (tid == 0) g_odata[blockIdx.x] = result;
}



__global__ void reduceKernelSum2(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i] : 0;

	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = getSum(sdata[tid], sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduceKernelSum3(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getSum(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = result = getSum(result, sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = result;
}
template <unsigned int blockSize>
__global__ void reduceKernelSum4(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getSum(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = result = getSum(result, sdata[tid + s]);
		}
		cg::sync(cta);
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		if (blockSize >= 64) result = getSum(result, sdata[tid + 32]);

		for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
		{
			result = getSum(result, tile32.shfl_down(result, offset));
		}
	}

	if (tid == 0) g_odata[blockIdx.x] = result;
}
template <unsigned int blockSize>
__global__ void reduceKernelSum5(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getSum(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = result = getSum(result, sdata[tid + 256]);
	}
	cg::sync(cta);


	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = result = getSum(result, sdata[tid + 128]);
	}
	cg::sync(cta);


	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = result = getSum(result, sdata[tid + 64]);
	}
	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		if (blockSize >= 64) result = getSum(result, sdata[tid + 32]);

		for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
		{
			result = getSum(result, tile32.shfl_down(result, offset));
		}
	}

	if (tid == 0) g_odata[blockIdx.x] = result;
}



template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T* g_idata, T* g_odata, unsigned int n, T (*fun)(T, T))
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

void reduce(int wichKernel, int type, int size, int threads, int blocks, double *d_idata, double *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = threads * sizeof(double);

	switch (type)
	{
	case MAXIMUM:
		switch (wichKernel)
		{
		case 2:
			reduceKernelMax2 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		case 3:
			reduceKernelMax3 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		case 4:
			switch (threads)
			{
			case 512:
				reduceKernelMax4<512> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 256:
				reduceKernelMax4<256> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 128:
				reduceKernelMax4<128> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 64:
				reduceKernelMax4<64> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 32:
				reduceKernelMax4<32> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 16:
				reduceKernelMax4<16> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  8:
				reduceKernelMax4<8> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  4:
				reduceKernelMax4<4> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  2:
				reduceKernelMax4<2> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  1:
				reduceKernelMax4<1> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;
			}
			break;

		case 5:
			switch (threads)
			{
			case 512:
				reduceKernelMax5<512> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 256:
				reduceKernelMax5<256> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 128:
				reduceKernelMax5<128> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 64:
				reduceKernelMax5<64> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 32:
				reduceKernelMax5<32> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 16:
				reduceKernelMax5<16> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  8:
				reduceKernelMax5<8> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  4:
				reduceKernelMax5<4> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  2:
				reduceKernelMax5<2> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  1:
				reduceKernelMax5<1> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;
			}
			break;

		default:
			throw;
			//break;
		}
		break;

	case SUMMATION:
		switch (wichKernel)
		{
		case 2:
			reduceKernelSum2 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		case 3:
			reduceKernelSum3 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		case 4:
			switch (threads)
			{
			case 512:
				reduceKernelSum4<512> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 256:
				reduceKernelSum4<256> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 128:
				reduceKernelSum4<128> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 64:
				reduceKernelSum4<64> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 32:
				reduceKernelSum4<32> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 16:
				reduceKernelSum4<16> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  8:
				reduceKernelSum4<8> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  4:
				reduceKernelSum4<4> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  2:
				reduceKernelSum4<2> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  1:
				reduceKernelSum4<1> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;
			}
			break;
		case 5:
			switch (threads)
			{
			case 512:
				reduceKernelSum5<512> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 256:
				reduceKernelSum5<256> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 128:
				reduceKernelSum5<128> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 64:
				reduceKernelSum5<64> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 32:
				reduceKernelSum5<32> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case 16:
				reduceKernelSum5<16> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  8:
				reduceKernelSum5<8> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  4:
				reduceKernelSum5<4> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  2:
				reduceKernelSum5<2> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;

			case  1:
				reduceKernelSum5<1> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
				break;
			}
			break;
		default:
			throw;
			//break;
		}
		break;

	default:
		//throw;
		break;
	}
}



template <typename T>
void reduce_v2(int size, int threads, int blocks, T(*fun)(T, T), T* d_idata, T* d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
                reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
                reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
                reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
                reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
                reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
                reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
                reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
                reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  1:
                reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 256:
                reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 128:
                reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 64:
                reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 32:
                reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case 16:
                reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  8:
                reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  4:
                reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  2:
                reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;

            case  1:
                reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, fun);
                break;
        }
    }
}



