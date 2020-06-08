#include "stdafx.h"

void systemEquCuda_3D::evaluate()
{
	double t = tau, dt;

	//Grid.calculateRhoK();
	//Grid.calculateOmega();
	//distr.setDistributionFunctionAsync(Grid.get_time(), Grid.get_rhoK_ptr(), Grid.get_omega_ptr());

	while (t >= (dt = Grid.get_dt(precision)) ) {
		evlulate_step(dt);
		
		//if (distr.isDistributionFunctionReady())
		//{
		//	Grid.calculateRhoK();
		//	Grid.calculateOmega();
		//	distr.setDistributionFunctionAsync(Grid.get_time(), Grid.get_rhoK_ptr(), Grid.get_omega_ptr());
		//}

		t -= dt;
	}

	if (t < dt && t > 0) {
		evlulate_step(t);
	}

	//distr.waitUntilAsyncEnd();

}
