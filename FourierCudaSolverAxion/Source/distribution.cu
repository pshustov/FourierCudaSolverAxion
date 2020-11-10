//#include "stdafx.h"
#include <device_launch_parameters.h>

#include "distribution.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ real getOmega(real lam, real g, real f2mean, real k_sqr)
{
	return sqrt(1 + k_sqr + 3 * lam * f2mean + 15 * g * f2mean * f2mean);
}

__global__ void setQquad(cudaCVector3Dev Q) 
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;
	
	size_t N1 = Q.getN1(), N2 = Q.getN2(), N3 = Q.getN3();

	if (i < N1 && j < N2 && k < N3)
	{
		size_t ind = (i * N2 + j) * N3 + k;
		Q(ind) = Q(ind).absSqr();
		//if (k == 0) {
		//	Q(ind) *= Q(ind).get_conj();
		//}
		//else {
		//	Q(ind) *= 2 * Q(ind).get_conj();
		//}
	}

}

__global__ void kernelCalculateDistrFun(real lam, real g, real f2mean, cudaRVector3Dev kSqr, cudaCVector3Dev Q, cudaCVector3Dev P) 
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;

	size_t N1 = kSqr.getN1(), N2 = kSqr.getN2(), N3 = kSqr.getN3();
	
	if (i < N1 && j < N2 && k < N3)
	{
		size_t ind = (i * N2 + j) * N3 + k;
		real omega = getOmega(lam, g, f2mean, kSqr(ind));
		real m = 1 + kSqr(ind) + 1.5 * lam * f2mean + 5 * g * f2mean * f2mean;

		if (k == 0) {
			P(ind) = 0.5 * (P(ind).absSqr() + m * Q(ind)) / omega;
			Q(ind) = sqrt(kSqr(ind))* P(ind);
		}
		else {
			P(ind) = (P(ind).absSqr() + m * Q(ind)) / omega;
			Q(ind) = sqrt(kSqr(ind))* P(ind);
		}
	}
}

__global__ void kernelSetKInds(int numberOfBins, real kMax, cudaRVector3Dev kSqr, cudaVector3Dev<unsigned int> kInds)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < kSqr.getSize())
	{
		real t = sqrt(kSqr(i)) / kMax;
		if (t < 1) {
			kInds(i) = numberOfBins * t;
		}
		else {
			kInds(i) = numberOfBins;
		}
	}
}

__global__ void kernelSetDenominators(cudaVector3Dev<unsigned int> kInds, cudaRVectorDev denoms)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < kInds.getSize())
	{
		atomicAdd(&denoms(kInds(i)), 1);
	}
}

__global__ void kernelSetZero(cudaRVectorDev data)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < data.getN())
	{
		data(i) = 0;
	}
}

__global__ void kernelCalculateDistrLin(cudaVector3Dev<unsigned int> kInds, cudaRVectorDev distrLin, cudaCVector3Dev data)
{
	cg::thread_block thisBlock = cg::this_thread_block();
	
	extern __shared__ real distrShared[];

	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;

	size_t N1 = data.getN1(), N2 = data.getN2(), N3 = data.getN3();
	
	size_t ind = (i * N2 + j) * N3 + k;
	size_t indThread = (threadIdx.x * blockDim.y + threadIdx.y) * blockDim.z + threadIdx.z;

	int numberOfBins = distrLin.getN();

	if (i < N1 && j < N2 && k < N3)
	{
		if (indThread < numberOfBins)
		{
			distrShared[indThread] = 0;
		}
		
		thisBlock.sync();

		if (k == 0) {
			atomicAdd(&distrShared[kInds(ind)], data(ind).get_real());
		}
		else {
			atomicAdd(&distrShared[kInds(ind)], 2 * data(ind).get_real());
		}
		
		thisBlock.sync();

		if (indThread < numberOfBins)
		{
			atomicAdd(&distrLin(indThread), distrShared[indThread]);
		}
	}	
}

__global__ void kernelDivide(cudaRVectorDev distrLin, cudaRVectorDev denom)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < distrLin.getN())
	{
		distrLin(i) /= denom(i);
	}
}


void Distribution::setupDistribution(cudaGrid_3D& Grid)
{
	mode = (isLoad ? std::ios_base::app : std::ios_base::out);
	outFile.open("outNumberAndMomentum.txt", mode);
	outFileDistr.open("outDistributionLin.txt", mode);

	time = Grid.get_time();
	lam = Grid.get_lambda();
	g = Grid.get_g();
	f2mean = 0;
	volume = Grid.getVolume();
	k_sqr = Grid.get_k_sqr();
	Q = Grid.get_Q();
	P = Grid.get_P();

	size_t Bx = 16, By = 16, Bz = 1;
	block3 = dim3(static_cast<unsigned int>(Bx), static_cast<unsigned int>(By), static_cast<unsigned int>(Bz));
	grid3Red = dim3(static_cast<unsigned int>((Q.getN1() + Bx - 1) / Bx), static_cast<unsigned int>((Q.getN2() + By - 1) / By), static_cast<unsigned int>((Q.getN3() + Bz - 1) / Bz));


	numberOfBins = 64;
	kInds.set(Q.getN1(), Q.getN2(), Q.getN3());
	distrLin.set(numberOfBins + 1);
	denominators.set(numberOfBins + 1);
	distrLinHost.set(numberOfBins + 1);

	numberOfParticles = 0;
	meanMomentum = 0;

	//cudaStreamCreate(&streamDistrib);
	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	cudaStreamCreateWithPriority(&streamDistrib, cudaStreamNonBlocking, priority_low);

	
	real k1 = Ma_PI * Grid.getN1() / Grid.getL1();
	real k2 = Ma_PI * Grid.getN2() / Grid.getL2();
	real k3 = Ma_PI * Grid.getN3() / Grid.getL3();
	real kMax = (kMax = (k1 < k2 ? k1 : k2)) < k3 ? kMax : k3;

	int blockSize = 256;
	dim3 blockT = blockSize;
	dim3 gridT = static_cast<unsigned int>((k_sqr.getSize() + blockSize - 1) / blockSize);
	kernelSetKInds<<< gridT, blockT, 0, streamDistrib >>>(numberOfBins, kMax, k_sqr, kInds);
	cudaStreamSynchronize(streamDistrib);

	blockSize = 32;
	blockT = blockSize;
	gridT = static_cast<unsigned int>((denominators.getN() + blockSize - 1) / blockSize);
	kernelSetZero<<< gridT, blockT, 0, streamDistrib >>>(denominators);

	blockSize = 256;
	blockT = blockSize;
	gridT = static_cast<unsigned int>((kInds.getSize() + blockSize - 1) / blockSize);
	kernelSetDenominators<<< gridT, blockT, 0, streamDistrib >>>(kInds, denominators);

	cudaStreamSynchronize(streamDistrib);

	//std::ofstream outDen("outDenom.txt");
	//RVector P = denominators;
	//for (size_t i = 0; i < P.getN() - 1; i++)
	//{
	//	outDen << P(i) << "\t";
	//}
	//outDen << P(P.getN() - 1);
	//outDen.close();

	if (~isLoad)
	{
		outFileDistr << numberOfBins;
		for (int i = 0; i < numberOfBins; i++)
		{
			outFileDistr << "\t" << (i + 1) * kMax / numberOfBins;
		}
		outFileDistr << std::endl;
	}
}

void Distribution::calculate()
{
	setQquad<<< grid3Red, block3, 0, streamDistrib >>>(Q);
	//cudaStreamSynchronize(streamDistrib);

	complex f2m = Q.getSum(streamDistrib).get_real() / (volume * volume);
	f2mean = f2m.get_real();

	if ((1 + 3 * lam * f2mean + 15 * g * f2mean * f2mean) < 0) {
		numberOfParticles = -1;
		meanMomentum = -1;

		if (!isAlarmed) {
			std::cout << "!!! Unstable condition !!!" << std::endl;
			isAlarmed = true;

			int blockSize = 32;
			dim3 blockT = blockSize;
			dim3 gridT = static_cast<unsigned int>((distrLin.getN() + blockSize - 1) / blockSize);
			kernelSetZero<<< gridT, blockT, 0, streamDistrib >>>(distrLin);
			cudaStreamSynchronize(streamDistrib);
			distrLinHost = distrLin;
		}
	}
	else
	{
		kernelCalculateDistrFun<<< grid3Red, block3, 0, streamDistrib >>>(lam, g, f2mean, k_sqr, Q, P);
		//cudaStreamSynchronize(streamDistrib);

		numberOfParticles = P.getSum(streamDistrib).get_real() / volume;
		meanMomentum = Q.getSum(streamDistrib).get_real() / (volume * numberOfParticles);

		int blockSize = 32;
		dim3 blockT = blockSize;
		dim3 gridT = static_cast<unsigned int>((distrLin.getN() + blockSize - 1) / blockSize);
		kernelSetZero<<< gridT, blockT, 0, streamDistrib >>>(distrLin);
		kernelCalculateDistrLin<<< grid3Red, block3, (numberOfBins+1)*sizeof(real), streamDistrib >>> (kInds, distrLin, P);
		kernelDivide<<< gridT, blockT, 0, streamDistrib >>>(distrLin, denominators);
		cudaStreamSynchronize(streamDistrib);
		distrLinHost = distrLin;

		isAlarmed = false;
	}


	outFile << time << '\t' << numberOfParticles << '\t' << meanMomentum << std::endl;
	outFileDistr << time;
	for (int i = 0; i < numberOfBins; i++)
	{
		outFileDistr << "\t" << distrLinHost(i);
	}
	outFileDistr << std::endl;

}

