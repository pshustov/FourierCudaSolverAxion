#pragma once

class equationsAxionSymplectic_3D
{
public:
	equationsAxionSymplectic_3D(cudaGrid_3D& _Grid);
	~equationsAxionSymplectic_3D() { cudaStreamDestroy(stream); }

	void equationCuda(const double dt);
	void getNonlin_Phi4_Phi6();

	void setCudaStream(cudaStream_t& _stream) { stream = _stream; }

private:
	int N1, N2, N3, N3red, N, Nred;
	dim3 block, grid, gridRed;
	
	double normT;

	cudaStream_t stream;
	cudaGrid_3D& Grid;
};


