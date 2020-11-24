#pragma once
#include <future>
#include <fstream>
#include "cudaGrid.h"

class Distribution
{
public:
	Distribution(bool _isLoad) : isLoad(_isLoad) { }
	~Distribution() { outFile.close(); cudaStreamDestroy(streamDistrib); }

	void setupDistribution(cudaGrid_3D& Grid);

	bool isDistributionFunctionReady()
	{
		return distributionFunctionFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	void calculate();

	void calculateAsync(cudaGrid_3D& Grid)
	{
		cudaStreamSynchronize(Grid.get_mainStream());
		time = Grid.get_time();
		Q = Grid.get_Q();
		P = Grid.get_P();

		distributionFunctionFuture = std::async(std::launch::async, &Distribution::calculate, this);
	}

	void calculateSync(cudaGrid_3D& Grid)
	{
		time = Grid.get_time();
		Q = Grid.get_Q();
		P = Grid.get_P();

		calculate();
	}

	void waitUntilAsyncEnd()
	{
		distributionFunctionFuture.get();
	}

	real getNumberOfParticles() const { return numberOfParticles; }
	real getMeanMomentum() const { return meanMomentum; }
	real getTau() const {
		real p2 = meanMomentum * meanMomentum;
		real lam2 = lam * lam;
		real v = meanMomentum / sqrt(1. + p2);
		real sigma = 9. * lam2 / (32. * M_PI * (1. + p2));		
		real f = 6. * M_PI * M_PI * numberOfParticles / (meanMomentum * meanMomentum * meanMomentum * volume);
		real n = numberOfParticles / volume;
		return 1 / (sigma * n * v * (1 + f));
	}
	real getInstability() const { return instability; }

private:
	
	bool isLoad;
	std::ios_base::openmode mode;
	std::ofstream outFile;
	std::ofstream outFileDistr;

	real time;
	real lam, g;
	real f2mean;
	real volume;
	real instability;

	cudaRVector3 k_sqr;
	cudaCVector3 Q, P;

	dim3 block3;
	dim3 grid3Red;

	int numberOfBins;
	cudaVector3<unsigned int> kInds;
	cudaRVector distrLin;
	cudaRVector denominators;
	RVector distrLinHost;

	real numberOfParticles, meanMomentum;

	std::future<void> distributionFunctionFuture;

	bool isAlarmed = false;

	cudaStream_t streamDistrib;
};