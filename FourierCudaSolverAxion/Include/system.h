#pragma once
#include <fstream>
#include <chrono>
#include "cudaGrid.h"
#include "distribution.h"
#include "equations.h"
#include "interface.h"

#include <helper_cuda.h>

class systemEquCuda_3D
{
public:
	systemEquCuda_3D(Params params)
		: precision(params.precision()), tau(params.tau()), Grid(params.filename()), Equation(Grid), distr(params.isLoad())
	{
		if (params.isLoad())
		{
			loadParams();
		}
		else
		{
			Grid.set_lambda(params.lambda());
			Grid.set_g(params.g());
		}
		energy0 = getEnergy();
		energyPrev = energy0;

		std::ios_base::openmode mode = (params.isLoad() ? std::ios_base::app : std::ios_base::out);
		outMaxVal.open("outMaxVal.txt", mode);
		outMaxVal.precision(14);

		distr.setupDistribution(Grid);
		distr.calculateAsync(Grid);
		distr.waitUntilAsyncEnd();
		cudaDeviceSynchronize();
	}

	~systemEquCuda_3D() {
		outMaxVal.close();
	}

	void evaluate();

	void save();
	void loadParams(std::string filename = "saveParams.asv");

	void printingVTK(bool isVTKprinting = true);
	void printingMaxVal();

	real getTime() { return Grid.get_time(); }
	real getEnergy() { return energy = Grid.getEnergy(); }
	real getEnergyPrev() { return energyPrev = Grid.getEnergyPrev(); }
	real getDelta0() { return (getEnergy() - energy0) / energy0; }
	real getDelta1() { 
		getEnergyPrev();
		return (getEnergy() - energyPrev) / energyPrev; 
	}

private:
	real precision;
	real tau;
	cudaGrid_3D Grid;
	equationsAxionSymplectic_3D Equation;
	Distribution distr;

	real energy0, energyPrev, energy;

	std::ofstream outMaxVal;
	std::ofstream outVTK;


	void evlulate_step(const real _dt)
	{
		Equation.equationCuda(_dt);
		getLastCudaError("Error after step: ");
	}

};
