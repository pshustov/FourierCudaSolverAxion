#include "stdafx.h"

__device__ double getOmega(double lam, double g, double f2mean, double k_sqr)
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
		double m = 1 + kSqr(ind) + 1.5 * lam * f2mean + 5 * g * f2mean * f2mean;

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

__global__ void kernelCalculateDistrLin(cudaRVectorDev kBinsLin, cudaCVectorDev distrLin, cudaCVector3Dev kSqr, cudaCVector3Dev data)
{
	extern __shared__ complex distrShared[];

	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;

	size_t N1 = kSqr.getN1(), N2 = kSqr.getN2(), N3 = kSqr.getN3();

	if (i < N1 && j < N2 && k < N3)
	{
		if (k == 0) {
			
		}
		else {

		}
	}
}


void Distribution::setDistribution(cudaGrid_3D& Grid)
{
	outFile.open("outNumberAndMomentum.txt");

	time = Grid.get_time();
	lam = Grid.get_lambda();
	g = Grid.get_g();
	volume = Grid.getVolume();
	f2mean = 0;
	k_sqr = Grid.get_k_sqr();
	Q = Grid.get_Q();
	P = Grid.get_P();

	size_t Bx = 16, By = 16, Bz = 1;
	block3 = dim3(Bx, By, Bz);
	grid3Red = dim3((Q.getN1() + Bx - 1) / Bx, (Q.getN2() + By - 1) / By, (Q.getN3() + Bz - 1) / Bz);


	numberOfBins = 100;
	
	double k1 = Ma_PI * Grid.getN1() / Grid.getL1();
	double k2 = Ma_PI * Grid.getN2() / Grid.getL2();
	double k3 = Ma_PI * Grid.getN3() / Grid.getL3();
	double kMax = (kMax = (k1 < k2 ? k1 : k2)) < k3 ? kMax : k3;

	RVector kBinsLin_host(numberOfBins);
	for (int i = 0; i < numberOfBins; i++) {
		kBinsLin_host(i) = (i + 1) * kMax / numberOfBins;
	}
	kBinsLin = kBinsLin_host;

	cudaStreamCreate(&streamDistrib);
}

void Distribution::calculate()
{
	setQquad<<< grid3Red, block3, 0, streamDistrib >>>(Q);
	cudaStreamSynchronize(streamDistrib);

	complex f2m = Q.getSum(streamDistrib).real() / (volume * volume);
	f2mean = f2m.real();

	if ((1 + 3 * lam * f2mean + 15 * g * f2mean * f2mean) < 0) {
		numberOfParticles = -1;
		meanMomentum = -1;

		if (!isAlarmed) {
			std::cout << "!!! Unstable condition !!!" << std::endl;
			isAlarmed = true;
		}
	}
	else
	{
		kernelSetDistributionFunction<<< grid3Red, block3, 0, streamDistrib >>>(lam, g, f2mean, k_sqr, Q, P);
		cudaStreamSynchronize(streamDistrib);



		numberOfParticles = P.getSum(streamDistrib).real() / volume;
		meanMomentum = Q.getSum(streamDistrib).real() / (volume * numberOfParticles);

		isAlarmed = false;
	}

	outFile << time << '\t' << numberOfParticles << '\t' << meanMomentum << std::endl;
}

