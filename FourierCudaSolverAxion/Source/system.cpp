#include "stdafx.h"

double reductionMax(int size, double *inData);
double reductionSum(int size, double *inData);

void systemEquCuda_3D::evaluate()
{
	double t = tau, dt;

	while (t >= (dt = Grid.get_dt(precision)) ) {
		evlulate_step(dt);
		t -= dt;
	}

	if (t < dt && t > 0) {
		evlulate_step(t);
	}

	isEnergyCalculated = false;
}

double systemEquCuda_3D::get_energy()
{
	return Grid.getEnergy();
}

