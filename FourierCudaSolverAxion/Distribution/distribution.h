#pragma once
#include "stdafx.h"

//using namespace std;
using namespace std::chrono;

class Distribution
{
public:
	Distribution() {}
	~Distribution() { outFile.close(); cudaStreamDestroy(streamDistrib); }

	void setupDistribution(cudaGrid_3D& Grid);

	bool isDistributionFunctionReady()
	{
		return distributionFunctionFuture.wait_for(seconds(0)) == std::future_status::ready;
	}

	void calculate();

	void calculateAsync(cudaGrid_3D& Grid)
	{
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

	std::ofstream outFile;
	std::ofstream outFileDistr;

	double time;
	double lam, g;
	double f2mean;
	double volume;

	cudaRVector3 k_sqr;
	cudaCVector3 Q, P;

	dim3 block3;
	dim3 grid3Red;

	int numberOfBins;
	cudaVector3<unsigned __int8> kInds;
	cudaRVector distrLin;
	cudaRVector denominators;
	RVector distrLinHost;

	double numberOfParticles, meanMomentum;

	std::future<void> distributionFunctionFuture;

	bool isAlarmed = false;

	cudaStream_t streamDistrib;



};