#include "stdafx.h"

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
}
