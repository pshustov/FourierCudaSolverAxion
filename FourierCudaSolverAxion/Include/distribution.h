#pragma once
#include <future>
#include <fstream>
#include "cudaGrid.h"

class Distribution
{
public:
	Distribution() {}
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
		constexpr real C = 64. / (3. * 3.1415926535897932384626433832795);
		real p3 = meanMomentum * meanMomentum * meanMomentum;
		real L6 = volume * volume;
		real lam2 = lam * lam;
		real N2 = numberOfParticles * numberOfParticles;
		return C * p3 * L6 / (lam2 * N2);
	}

private:

	std::ofstream outFile;
	std::ofstream outFileDistr;

	real time;
	real lam, g;
	real f2mean;
	real volume;

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