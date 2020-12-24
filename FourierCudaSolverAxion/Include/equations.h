#pragma once

class equationsAxionSymplectic_3D
{
public:
	equationsAxionSymplectic_3D(cudaGrid_3D& _Grid);
	~equationsAxionSymplectic_3D() { cudaStreamDestroy(stream); cudaFree(dt_local); }

	void equationCuda(const real dt);
	void getNonlin_Phi4_Phi6();

	void setCudaStream(cudaStream_t& _stream) { stream = _stream; }
	void createGraph();

private:
	int N1, N2, N3, N3red, N, Nred;
	dim3 block, grid, gridRed;

	cudaStream_t stream;
	cudaGrid_3D& Grid;

	cudaGraph_t graph;
	cudaGraphExec_t graphExec;

	bool isGraphCreated;
	real* dt_local;
};


