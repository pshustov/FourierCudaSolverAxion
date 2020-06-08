#pragma once

#include "stdafx.h"


class equationsAxionSymplectic_3D
{
public:
	equationsAxionSymplectic_3D(cudaStream_t _stream = cudaStreamLegacy);
	~equationsAxionSymplectic_3D() { cudaStreamDestroy(stream); }

	void equationCuda(const double dt, cudaGrid_3D & Grid);
	void getNonlin_Phi4_Phi6(cudaGrid_3D & Grid);

	void setCudaStream(cudaStream_t& _stream) { stream = _stream; }

private:
	cudaStream_t stream;
	cudaGraph_t graph;

	const size_t N_sympectic = 4;
	double C[4];
	double D[4];
	double *Cdev;
	double *Ddev;
};


