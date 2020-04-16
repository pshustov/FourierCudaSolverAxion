#pragma once

#include "stdafx.h"


class equationsAxionSymplectic_3D
{
public:
	equationsAxionSymplectic_3D();
	~equationsAxionSymplectic_3D() {}

	void equationCuda(const double dt, cudaGrid_3D & Grid);
	void getNonlin_Phi4_Phi6(cudaGrid_3D & Grid);

private:

	const size_t N_sympectic = 4;
	double C[4];
	double D[4];
	double *Cdev;
	double *Ddev;
};
