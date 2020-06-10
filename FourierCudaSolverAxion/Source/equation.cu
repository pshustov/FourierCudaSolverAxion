#include "stdafx.h"


__global__ void kernalStepSymplectic41_v2(const double dt, const double normT, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.size())
	{
		P(i) -= 0.67560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i) * normT) * dt;
		Q(i) += 1.3512071919596576340476878089715 * P(i) * dt;
	}
}
__global__ void kernalStepSymplectic42_v2(const double dt, const double normT, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.size())
	{
		P(i) -= -0.17560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i) * normT) * dt;
		Q(i) += -1.702414383919315268095375617943 * P(i) * dt;
	}
}
__global__ void kernalStepSymplectic43_v2(const double dt, const double normT, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.size())
	{
		P(i) -= -0.17560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i) * normT) * dt;
		Q(i) += 1.3512071919596576340476878089715 * P(i) * dt;
	}
}
__global__ void kernalStepSymplectic44_v2(const double dt, const double normT, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.size())
	{
		P(i) -= 0.67560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i) * normT) * dt;
	}
}

__global__ void kernel_Phi4_Phi6_v2(const int N, const double L, const double lambda, const double g, cudaRVector3Dev q, cudaRVector3Dev t)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double f = q(i) / L;
	if (i < N)
	{
		t(i) = f * f * f * (lambda + g * f * f);
	}
}

equationsAxionSymplectic_3D::equationsAxionSymplectic_3D(cudaStream_t _stream)
{
	stream = _stream;
}

void equationsAxionSymplectic_3D::equationCuda(const double dt, cudaGrid_3D& Grid)
{
	int N1 = (int)Grid.getN1();
	int N2 = (int)Grid.getN2();
	int N3red = (int)Grid.getN3red();
	int Nred = N1 * N2 * N3red;

	dim3 block(BLOCK_SIZE);
	dim3 grid((Nred + BLOCK_SIZE + 1) / BLOCK_SIZE);

	double normT = Grid.getVolume() / Grid.size();
	
	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic41_v2<<<grid, block, 0, stream>>>(dt, normT, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
		
	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic42_v2<<<grid, block, 0, stream>>>(dt, normT, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	
	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic43_v2<<<grid, block, 0, stream>>>(dt, normT, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	
	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic44_v2<<<grid, block, 0, stream>>>(dt, normT, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	
	cudaStreamSynchronize(stream);
	Grid.setSmthChanged();
	
	Grid.timestep(dt);
}

void equationsAxionSymplectic_3D::getNonlin_Phi4_Phi6(cudaGrid_3D& Grid)
{
	int N1 = (int)Grid.getN1();
	int N2 = (int)Grid.getN2();
	int N3 = (int)Grid.getN3();
	int N3red = (int)Grid.getN3red();
	int N = N1 * N2 * N3;

	dim3 block(BLOCK_SIZE);
	dim3 grid((N + BLOCK_SIZE + 1) / BLOCK_SIZE);

	bool isNormed = false;
	Grid.ifftQ(isNormed, true);
	kernel_Phi4_Phi6_v2<<<grid, block, 0, stream>>>(N, Grid.getVolume(), Grid.get_lambda(), Grid.get_g(), Grid.get_q(), Grid.get_t());
	Grid.doFFTforward(Grid.get_t(), Grid.get_T(), false);
}
