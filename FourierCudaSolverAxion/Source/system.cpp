#include "stdafx.h"

void systemEquCuda_3D::evaluate()
{
	double t = tau, dt;

	//distr.calculateNumberAndMomentumAsync(Grid);

	int countIn = 0, countOut = 0;

	while (t >= (dt = Grid.get_dt(precision)) ) {
		evlulate_step(dt);
		++countOut;
		/*if (distr.isDistributionFunctionReady())
		{
			++countIn;
			distr.calculateNumberAndMomentumAsync(Grid);
		}*/

		t -= dt;
	}

	if (t < dt && t > 0) {
		evlulate_step(t);
	}

	//std::cout << "Current n = " << distr.getNumberOfParticles() << ", p = " << distr.getMeanMomentum() << ", cIn/cOut = " << (double)countIn / (double)countOut << std::endl;
}