#pragma once
#include "stdafx.h"

using namespace std;
using namespace std::chrono;

class Distribution
{
public:
	Distribution(cudaGrid_3D &Grid);
	~Distribution() { fDistrOut.close(); }

	bool isDistributionFunctionReady()
	{
		return distributionFunctionFuture.wait_for(seconds(0)) == future_status::ready;
	}

	void setDistributionFunction(const complex* rhoKcuda, const double* omegaCuda);
	void setDistributionFunction_v2();

	void setDistributionFunctionAsync(const double _time, const complex *rhoK, const double *omega)
	{
		time = _time;
		distributionFunctionFuture = std::async(std::launch::async, &Distribution::setDistributionFunction, this, rhoK, omega);
	}
	void setDistributionFunctionAsync_v2(cudaGrid_3D& Grid)
	{
		time = Grid.get_time();
		Q = Grid.get_Q();
		P = Grid.get_P();

		distributionFunctionFuture = std::async(std::launch::async, &Distribution::setDistributionFunction_v2, this);
	}

	void waitUntilAsyncEnd()
	{
		distributionFunctionFuture.get();
	}


private:
	static const int Nf = 64;
	static const int powNf = 6;

	int N;
	double kappa;
	double boundariesSqr[Nf + 1];
	double time;

	CVector3 Q, P;

	CVector3 rhoK;
	RVector3 omega;

	CVector f;
	std::ofstream fDistrOut;
	std::future<void> distributionFunctionFuture;

	void printDistr()
	{
		fDistrOut << time;
		for (int i = 0; i < Nf; i++)
		{
			if (std::isnan(f(i).abs())) {
				fDistrOut << "\t" << 0;
			}
			else {
				fDistrOut << "\t" << f(i).abs();
			}
		}
		fDistrOut << std::endl;
	}
};