#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudaGrid.h"
#include "equations.h"

#define GRAPH

__global__ void kernalStepSymplectic41_v2(const real dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.getSize())
	{
		P(i) -= 0.67560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt;
		Q(i) += 1.3512071919596576340476878089715 * P(i) * dt;
	}
}
__global__ void kernalStepSymplectic42_v2(const real dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.getSize())
	{
		P(i) -= -0.17560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt;
		Q(i) += -1.702414383919315268095375617943 * P(i) * dt;
	}
}
__global__ void kernalStepSymplectic43_v2(const real dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.getSize())
	{
		P(i) -= -0.17560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt;
		Q(i) += 1.3512071919596576340476878089715 * P(i) * dt;
	}
}
__global__ void kernalStepSymplectic44_v2(const real dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < Q.getSize())
	{
		P(i) -= 0.67560359597982881702384390448573 * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt;
	}
}


template <unsigned int stepn>
__global__ void kernalStepSymplectic8_v3(const real dt, cudaRVector3Dev k_sqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	real C;
	if (i < Q.getSize())
	{
		switch (stepn)
		{
		case 1:
			C = 0.74167036435061295344822780;
		case 2:
			C = -0.40910082580003159399730010;
		case 3:
			C = 0.19075471029623837995387626;
		case 4:
			C = -0.57386247111608226665638773;
		case 5:
			C = 0.29906418130365592384446354;
		case 6:
			C = 0.33462491824529818378495798;
		case 7:
			C = 0.31529309239676659663205666;
		case 8:
			C = -0.79688793935291635401978884;
		case 9:
			C = 0.31529309239676659663205666;
		case 10:
			C = 0.33462491824529818378495798;
		case 11:
			C = 0.29906418130365592384446354;
		case 12:
			C = -0.57386247111608226665638773;
		case 13:
			C = 0.19075471029623837995387626;
		case 14:
			C = -0.40910082580003159399730010;
		case 15:
			C = 0.74167036435061295344822780;
		default:
			break;
		}
		complex Pi = P(i);		
		P(i) -= C * ((1 + k_sqr(i)) * Q(i) + T(i)) * dt;
		Q(i) += C * Pi * dt;
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

	isGraphCreated = false;
}

void equationsAxionSymplectic_3D::equationCuda(const real dt)
{

#ifdef GRAPH
	if (!isGraphCreated) {
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic41_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic42_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic43_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic44_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		cudaStreamEndCapture(stream, &graph);
		cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
		isGraphCreated = true;
	}
	cudaGraphLaunch(graphExec, stream);
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

void equationsAxionSymplectic_3D::equationCuda_v2(const real dt)
{

#ifdef _WIN64

	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<1> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<2> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<3> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<4> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<5> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<6> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<7> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<8> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<9> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<10> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<11> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<12> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<13> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<14> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());
	getNonlin_Phi4_Phi6();
	kernalStepSymplectic8_v3<15> << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

#endif // _WIN64

#ifdef __linux__
	if (!isGraphCreated) {
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic41_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic42_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic43_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		getNonlin_Phi4_Phi6();
		kernalStepSymplectic44_v2 << <gridRed, block, 0, stream >> > (dt, Grid.get_k_sqr(), Grid.get_Q(), Grid.get_P(), Grid.get_T());

		cudaStreamEndCapture(stream, &graph);
		cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
		isGraphCreated = true;
	}
	cudaGraphLaunch(graphExec, stream);
#endif // __linux__

	Grid.setSmthChanged();
	Grid.timestep(dt);
}


void equationsAxionSymplectic_3D::getNonlin_Phi4_Phi6()
{
	Grid.doFFTinverce(Grid.get_Q(), Grid.get_q());
	kernel_Phi4_Phi6_v2<<<grid, block, 0, stream>>>(N, Grid.getVolume(), Grid.get_lambda(), Grid.get_g(), Grid.get_q(), Grid.get_t());
	Grid.doFFTforward(Grid.get_t(), Grid.get_T());
}

void equationsAxionSymplectic_3D::createGraph()
{
}
