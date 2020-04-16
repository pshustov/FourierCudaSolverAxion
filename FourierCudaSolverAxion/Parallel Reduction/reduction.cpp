#include "stdafx.h"
#include "reduction.h"

void reduce(int witchKernel, int type, int size, int threads, int blocks, double *d_idata, double *d_odata);

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

bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}

void getNumBlocksAndThreads(int whichKernel, int n, int maxThreads, int &blocks, int &threads)
{
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	if (whichKernel < 3)
	{
		threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
		blocks = (n + threads - 1) / threads;
	}
	else
	{
		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
	}

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

double reductionMax(int size, double *inData)
{
	int witchKernel = 5;
	int cpuFinalThreshold = 256;
	int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(witchKernel, size, maxThreads, blocks, threads);

	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));

	reduce(witchKernel, MAXIMUM, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(witchKernel, s, maxThreads, blocks, threads);
		reduce(witchKernel, MAXIMUM, s, threads, blocks, inData_dev, outData_dev);

		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = outData_host[0];
	for (size_t i = 1; i < s; i++)
	{
		result = result > outData_host[i] ? result : outData_host[i];
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}

double reductionSum(int size, double *inData)
{
	int witchKernel = 5;
	int cpuFinalThreshold = 256;
	int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(witchKernel, size, maxThreads, blocks, threads);


	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));

	reduce(witchKernel, SUMMATION, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(2, s, maxThreads, blocks, threads);
		reduce(witchKernel, SUMMATION, s, threads, blocks, inData_dev, outData_dev);

		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = 0;
	for (size_t i = 0; i < s; i++)
	{
		result += outData_host[i];
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}

//
//double reductionSigma2(int size, double *inData)
//{
//
//	int cpuFinalThreshold = 256;
//	int maxThreads = 256;
//
//	if (!isPow2(size)) throw;
//
//	int blocks = 0, threads = 0;
//	getNumBlocksAndThreads(6, size, maxThreads, blocks, threads);
//
//
//	double *inData_dev = NULL;
//	double *outData_dev = NULL;
//
//	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
//	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));
//
//	reduce(6, SIGMA2, size, threads, blocks, inData, outData_dev);
//	cudaDeviceSynchronize();
//
//	int s = blocks;
//	while (s > cpuFinalThreshold)
//	{
//		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);
//
//		getNumBlocksAndThreads(6, s, maxThreads, blocks, threads);
//		reduce(6, MEAN, s, threads, blocks, inData_dev, outData_dev);
//
//		s = blocks;
//	}
//
//	double *outData_host;
//	outData_host = (double*)malloc(s * sizeof(double));
//	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);
//
//	double result = 0;
//	for (size_t i = 0; i < s; i++)
//	{
//		result += outData_host[i];
//	}
//	result /= s;
//
//	cudaFree(inData_dev);
//	cudaFree(outData_dev);
//	free(outData_host);
//
//	return result;
//}
//
//double reductionSigma4(int size, double *inData)
//{
//
//	int cpuFinalThreshold = 256;
//	int maxThreads = 256;
//
//	if (!isPow2(size)) throw;
//
//	int blocks = 0, threads = 0;
//	getNumBlocksAndThreads(6, size, maxThreads, blocks, threads);
//
//
//	double *inData_dev = NULL;
//	double *outData_dev = NULL;
//
//	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
//	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));
//
//	reduce(6, SIGMA4, size, threads, blocks, inData, outData_dev);
//	cudaDeviceSynchronize();
//
//	int s = blocks;
//	while (s > cpuFinalThreshold)
//	{
//		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);
//
//		getNumBlocksAndThreads(6, s, maxThreads, blocks, threads);
//		reduce(6, MEAN, s, threads, blocks, inData_dev, outData_dev);
//
//		s = blocks;
//	}
//
//	double *outData_host;
//	outData_host = (double*)malloc(s * sizeof(double));
//	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);
//
//	double result = 0;
//	for (size_t i = 0; i < s; i++)
//	{
//		result += outData_host[i];
//	}
//	result /= s;
//
//	cudaFree(inData_dev);
//	cudaFree(outData_dev);
//	free(outData_host);
//
//	return result;
//}
