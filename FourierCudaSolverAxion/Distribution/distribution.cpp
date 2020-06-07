#include "stdafx.h"

__device__ double getOmega(double lam, double g, double f2mean, double k_sqr)
{
	return sqrt(1 + k_sqr + 3 * lam * f2mean + 15 * g * f2mean * f2mean);
}

__global__ void kernelPrepare(cudaCVector3Dev Q)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;
	
	size_t N1 = Q.getN1(), N2 = Q.getN2(), N3 = Q.getN3();

	if (i < N1 && j < N2 && k < N3 && k != 0)
	{
		size_t ind = (i * N2 + j) * N3 + k;
		Q(ind) *= 2;
	}
}

__global__ void kernelSetDistributionFunction(double lam, double g, double f2mean, cudaRVector3Dev kSqr, cudaCVector3Dev Q, cudaCVector3Dev P) 
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;

	size_t N1 = kSqr.getN1(), N2 = kSqr.getN2(), N3 = kSqr.getN3();
	
	if (i < N1 && j < N2 && k < N3)
	{
		size_t ind = (i * N2 + j) * N3 + k;
		double omega = getOmega(lam, g, f2mean, kSqr(ind));
		if (k == 0) {
			double m = 1 + kSqr(ind) + 1.5 * f2mean + 5 * g * f2mean * f2mean;
			P(ind) = 0.5 * (P(ind) + m * Q(ind)) / omega;
		}
		else {
			double m = 1 + kSqr(ind) + 1.5 * f2mean + 5 * g * f2mean * f2mean;
			P(ind) = (P(ind) + m * Q(ind)) / omega;
		}
	}
}


int inWhichInterval(const unsigned int N, const unsigned leftPowN, const double a, const double* b)
{
	if (a >= b[N] || a < b[0]) {
		return -1;
	}

	int k = N;
	int l = N;

	for (unsigned int i = 0; i < leftPowN; i++)
	{
		l >>= 1;
		if (N & (1 << i))
		{
			if (a < b[k - l])
			{
				if (a < b[k - l - 1])
				{
					k -= l + 1;
				}
				else
				{
					return k - l - 1;
				}
			}
		}
		else
		{
			if (a < b[k - l])
			{
				k -= l;
			}
		}
	}
	throw;
}


Distribution::Distribution(cudaGrid_3D & Grid)
{
	outFile.open("outNumberAndMomentum.txt");

	time	= Grid.get_time();
	lam		= Grid.get_lambda();
	g		= Grid.get_g();
	f2mean	= 0;
	k_sqr	= Grid.get_k_sqr();
	Q		= Grid.get_Q();
	P		= Grid.get_P();
}


void Distribution::calculateNumberAndMomentum()
{
	size_t Bx = 16, By = 8, Bz = 1;
	dim3 block(Bx, By, Bz);
	dim3 grid((Q.getN1() + Bx - 1) / Bx, (Q.getN2() + By - 1) / By, (Q.getN3() + Bz - 1) / Bz);

	kernelPrepare<<<grid,block>>>(Q);
	cudaDeviceSynchronize();

	//kernelSetDistributionFunction(lam, g, )
}

