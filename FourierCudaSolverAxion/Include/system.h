#pragma once
#include <fstream>
#include "cudaGrid.h"
#include "distribution.h"
#include "equations.h"

class systemEquCuda_3D
{
public:
	systemEquCuda_3D(std::string filename, double _precision, double _tau, double _lambda = 0, double _g = 0, bool isLoadParams = false)
		: precision(_precision), tau(_tau), Grid(filename), Equation(Grid)
	{
		if (isLoadParams)
		{
			loadParams();
		}
		else
		{
			Grid.set_lambda(_lambda);
			Grid.set_g(_g);
		}
		energy0 = getEnergy();

		outMaxVal.open("outMaxVal.txt");
		outMaxVal.precision(14);

		distr.setupDistribution(Grid);
	}

	~systemEquCuda_3D() {
		outMaxVal.close();
	}

	void evaluate();

	void save();
	void loadParams(std::string filename = "saveParams.asv");

	void printingVTK(bool isVTKprinting = true);
	void printingMaxVal();

	double getTime() { return Grid.get_time(); }
	double getEnergy() { return energy = Grid.getEnergy(); }
	double getDelta() { return (getEnergy() - energy0) / energy0; }

private:
	double precision;
	double tau;
	cudaGrid_3D Grid;
	equationsAxionSymplectic_3D Equation;
	Distribution distr;

	double energy0, energy;

	std::ofstream outMaxVal;
	std::ofstream outVTK;


	void evlulate_step(const double _dt)
	{
		Equation.equationCuda(_dt);
	}

};
