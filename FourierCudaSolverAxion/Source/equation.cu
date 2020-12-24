﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudaGrid.h"
#include "equations.h"

#include <helper_cuda.h>

#define GRAPH

__global__ void kernalStepSymplectic41_v2(real* dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < Q.getSize())
	{
		P(i) -= 0.67560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt[0];
		Q(i) += 1.3512071919596576340476878089715 * P(i) * dt[0];
	}
}
__global__ void kernalStepSymplectic42_v2(real* dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.getSize())
	{
		P(i) -= -0.17560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt[0];
		Q(i) += -1.702414383919315268095375617943 * P(i) * dt[0];
	}
}
__global__ void kernalStepSymplectic43_v2(real* dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.getSize())
	{
		P(i) -= -0.17560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt[0];
		Q(i) += 1.3512071919596576340476878089715 * P(i) * dt[0];
	}
}
__global__ void kernalStepSymplectic44_v2(real* dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.getSize())
	{
		P(i) -= 0.67560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt[0];
	}
}

__global__ void kernel_Phi4_Phi6_v2(const int N, const real L, const real lambda, const real g, cudaRVector3Dev q, cudaRVector3Dev t)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	real f = q(i);// / L;
	if (i < N)
	{
		t(i) = f * f * f * (lambda + g * f * f);
	}
}

equationsAxionSymplectic_3D::equationsAxionSymplectic_3D(cudaGrid_3D& _Grid) : Grid(_Grid)
{
	stream = Grid.get_mainStream();

	N1		= (int)Grid.getN1();
	N2		= (int)Grid.getN2();
	N3		= (int)Grid.getN3();
	N3red	= (int)Grid.getN3red();
	N		= N1 * N2 * N3;
	Nred	= N1 * N2 * N3red;
	
	block	= dim3(BLOCK_SIZE);
	grid	= dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	gridRed = dim3((Nred + BLOCK_SIZE - 1) / BLOCK_SIZE);
	checkCudaErrors(cudaMalloc(&dt_local, sizeof(real)));

	isGraphCreated = false;
}

void equationsAxionSymplectic_3D::equationCuda(const real dt)
{
#ifdef GRAPH
	if (!isGraphCreated) {
		checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

		cudaMemcpyAsync(dt_local, &dt, sizeof(real), cudaMemcpyHostToDevice, stream);

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic41_v2 << <gridRed, block, 0, stream >> > (dt_local, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic42_v2 << <gridRed, block, 0, stream >> > (dt_local, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic43_v2 << <gridRed, block, 0, stream >> > (dt_local, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic44_v2 << <gridRed, block, 0, stream >> > (dt_local, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		checkCudaErrors(cudaStreamEndCapture(stream, &graph));
		checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
		isGraphCreated = true;
	}
	checkCudaErrors(cudaGraphLaunch(graphExec, stream));
	//checkCudaErrors(cudaStreamSynchronize(stream));

#else
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic41_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

	getNonlin_Phi4_Phi6();
	kernalStepSymplectic42_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

	getNonlin_Phi4_Phi6();
	kernalStepSymplectic43_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

	getNonlin_Phi4_Phi6();
	kernalStepSymplectic44_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

#endif // GRAPH

	Grid.setSmthChanged();
	Grid.timestep(dt);
}


void equationsAxionSymplectic_3D::getNonlin_Phi4_Phi6()
{
	Grid.doFFTinverce(Grid.get_Q(), Grid.get_q());
	kernel_Phi4_Phi6_v2<<<grid, block, 0, stream>>>(N, Grid.getVolume(), Grid.get_lambda(), Grid.get_g(), Grid.get_q(), Grid.get_t());
	Grid.doFFTforward(Grid.get_t(), Grid.get_T());
}
