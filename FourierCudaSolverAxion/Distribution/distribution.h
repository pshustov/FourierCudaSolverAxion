#pragma once
#include "stdafx.h"

//using namespace std;
using namespace std::chrono;

class Distribution
{
public:
	Distribution() {}
	~Distribution() { outFile.close(); cudaStreamDestroy(streamDistrib); }

	void setDistribution(cudaGrid_3D& Grid);

	bool isDistributionFunctionReady()
	{
		return distributionFunctionFuture.wait_for(seconds(0)) == std::future_status::ready;
	}

	void calculateNumberAndMomentum();

	void calculateNumberAndMomentumAsync(cudaGrid_3D& Grid)
	{
		time = Grid.get_time();
		Q = Grid.get_Q();
		P = Grid.get_P();

		distributionFunctionFuture = std::async(std::launch::async, &Distribution::calculateNumberAndMomentum, this);
	}

	void calculateNumberAndMomentumSync(cudaGrid_3D& Grid)
	{
		time = Grid.get_time();
		Q = Grid.get_Q();
		P = Grid.get_P();

		calculateNumberAndMomentum();
	}

	void waitUntilAsyncEnd()
	{
		distributionFunctionFuture.get();
	}

	double getNumberOfParticles() const { return numberOfParticles; }
	double getMeanMomentum() const { return meanMomentum; }
	double getTau() const {
		constexpr double C = 64. / (3. * 3.1415926535897932384626433832795);
		double p3 = meanMomentum * meanMomentum * meanMomentum;
		double L6 = volume * volume;
		double lam2 = lam * lam;
		double N2 = numberOfParticles * numberOfParticles;
		return C * p3 * L6 / (lam2 * N2);
	}

private:
	double time;

	double lam, g;
	double f2mean;
	cudaRVector3 k_sqr;
	cudaCVector3 Q, P;

	double volume;

	std::ofstream outFile;
	double numberOfParticles, meanMomentum;

	std::future<void> distributionFunctionFuture;

	bool isAlarmed = false;

	cudaStream_t streamDistrib;

	dim3 block3;
	dim3 grid3Red;

};