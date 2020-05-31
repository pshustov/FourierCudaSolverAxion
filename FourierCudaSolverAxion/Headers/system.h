#pragma once

#include "stdafx.h"

class systemEquCuda_3D
{
public:
	systemEquCuda_3D(std::string filename, double _precision, double _tau, double _lambda = 0, double _g = 0, bool isLoadParams = false)
		: precision(_precision), tau(_tau), Grid(filename), distr(Grid)
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
		isEnergyCalculated = false;
		energy0 = get_energy();

		out_maxVal.open("out_maxVal.txt");
		out_maxVal.precision(14);
	}
	~systemEquCuda_3D() {
		out_maxVal.close();
	}

	void evaluate();

	void save()
	{
		std::ofstream fsave;
		
		fsave.open("saveGrid.asv");
		fsave.precision(5);
		Grid.save(fsave);
		fsave << "\n" << Grid.get_time() << std::endl;
		fsave.close();

		fsave.open("saveParams.asv");
		fsave.precision(10);
		fsave << Grid.get_g() << "\n" << Grid.get_lambda() << std::endl;

		fsave.close();

	}
	void loadParams(std::string filename = "saveParams.asv")
	{
		std::ifstream fload(filename);

		double _g, _lambda;

		fload >> _g;
		fload >> _lambda;

		Grid.set_lambda(_lambda);
		Grid.set_g(_g);
	}

	double get_time() { return Grid.get_time(); }
	double get_energy();
	double get_delta() { return (energy - energy0) / energy0; }

private:
	double precision;
	double tau;
	cudaGrid_3D Grid;
	equationsAxionSymplectic_3D Equation;
	Distribution distr;

	double energy0, energy;
	bool isEnergyCalculated;

	std::ofstream out_maxVal;

	void evlulate_step(const double _dt)
	{
		Equation.equationCuda(_dt, Grid);
	}

};