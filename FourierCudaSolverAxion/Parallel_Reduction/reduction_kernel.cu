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



