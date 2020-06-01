#include "stdafx.h"

template <class T>
void reduce(int size, int threads, int blocks, T(*fun)(T, T), T* d_idata, T* d_odata);

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

void getNumBlocksAndThreads(int n, int maxThreads, int &blocks, int &threads)
{
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);

	if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
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

	//T (*fun)(T, T) = funSum<T>;

	//__host__ __device__ T(*fun)(T, T)  = [](T A, T B) { return A + B; };

	reduce<T>(size, threads, blocks, fun, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(T), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce<T>(s, threads, blocks, fun, inData_dev, outData_dev);

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
template int reductionSum(int size, int* inData);
template float reductionSum(int size, float* inData);
template double reductionSum(int size, double* inData);
//template complex reductionSum(int size, complex* inData);