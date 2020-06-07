#pragma once
#include "stdafx.h"

//using namespace std;
using namespace std::chrono;

class Distribution
{
public:
	Distribution(cudaGrid_3D &Grid);
	~Distribution() { outFile.close(); }

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

	void waitUntilAsyncEnd()
	{
		distributionFunctionFuture.get();
	}


private:
	double time;

	double lam, g;
	double f2mean;
	cudaRVector3 k_sqr;
	cudaCVector3 Q, P;

	std::ofstream outFile;
	double numberOfParticles, meanMomentum;

	std::future<void> distributionFunctionFuture;

};