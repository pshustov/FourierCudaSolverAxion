#pragma once

#include "stdafx.h"


class equationsAxionSymplectic_3D
{
public:
	equationsAxionSymplectic_3D(cudaGrid_3D& _Grid, cudaStream_t& _stream);
	~equationsAxionSymplectic_3D() { cudaStreamDestroy(stream); }

	void equationCuda(const double dt);
	void getNonlin_Phi4_Phi6();

	void setCudaStream(cudaStream_t& _stream) { stream = _stream; }

	void makeGraph();

private:
	double normT;
	dim3 block, grid, gridRed;

	cudaStream_t& stream;
	cudaGrid_3D &Grid;

	cudaGraph_t graph;
	cudaGraphExec_t instance;
};


