#include "stdafx.h"


equationsAxionSymplectic_3D::equationsAxionSymplectic_3D()
{
	constexpr double twoToOneOverThree = 1.2599210498948731647672106072782;
	switch (N_sympectic)
	{
	case 1:
		C[0] = 1.0;
		D[0] = 1.0;
		break;

	case 2:
		C[0] = 0.0;
		C[1] = 1.0;

		D[0] = 0.5;
		D[1] = 0.5;
		break;

	case 3:
		C[0] = 1.0;
		C[1] = -2.0 / 3.0;
		C[2] = 2.0 / 3.0;

		D[0] = -1.0 / 24.0;
		D[1] = 3.0 / 4.0;
		D[2] = 7.0 / 24.0;
		break;

	case 4:

		C[0] = 1.0 / (2.0 * (2.0 - twoToOneOverThree));
		C[1] = (1.0 - twoToOneOverThree) / (2.0 * (2.0 - twoToOneOverThree));
		C[2] = C[1];
		C[3] = C[0];

		D[0] = 1.0 / (2.0 - twoToOneOverThree);
		D[1] = -twoToOneOverThree / (2.0 - twoToOneOverThree);
		D[2] = D[0];
		D[3] = 0;
		break;

	default:
		break;
	}

	cudaMalloc(&Cdev, N_sympectic * sizeof(double));
	cudaMemcpy(Cdev, &C, N_sympectic * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&Ddev, N_sympectic * sizeof(double));
	cudaMemcpy(Ddev, &D, N_sympectic * sizeof(double), cudaMemcpyHostToDevice);
}
void equationsAxionSymplectic_3D::equationCuda(const double dt, cudaGrid_3D& Grid)
{

	int N1 = (int)Grid.get_N1();
	int N2 = (int)Grid.get_N2();
	int N3red = (int)Grid.get_N3red();
	int Nred = N1 * N2 * N3red;

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil( (double)Nred / (double)BLOCK_SIZE));


	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic41<<<grid, block>>>(Nred, dt, Grid.get_k_sqr_ptr(), Grid.get_Q_ptr(), Grid.get_P_ptr(), Grid.get_T_ptr());
	Grid.setSmthChanged();
	cudaDeviceSynchronize();

	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic42<<<grid, block>>>(Nred, dt, Grid.get_k_sqr_ptr(), Grid.get_Q_ptr(), Grid.get_P_ptr(), Grid.get_T_ptr());
	Grid.setSmthChanged();
	cudaDeviceSynchronize();

	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic43<<<grid, block>>>(Nred, dt, Grid.get_k_sqr_ptr(), Grid.get_Q_ptr(), Grid.get_P_ptr(), Grid.get_T_ptr());
	Grid.setSmthChanged();
	cudaDeviceSynchronize();

	getNonlin_Phi4_Phi6(Grid);
	kernalStepSymplectic44<<<grid, block>>>(Nred, dt, Grid.get_k_sqr_ptr(), Grid.get_Q_ptr(), Grid.get_P_ptr(), Grid.get_T_ptr());
	Grid.setSmthChanged();
	cudaDeviceSynchronize();

	Grid.timestep(dt);
}

void equationsAxionSymplectic_3D::getNonlin_Phi4_Phi6(cudaGrid_3D & Grid)
{
	int N1 = (int)Grid.get_N1();
	int N2 = (int)Grid.get_N2();
	int N3 = (int)Grid.get_N3();
	int N3red = (int)Grid.get_N3red();
	int N = N1 * N2 * N3;


	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)N / (double)BLOCK_SIZE));

	Grid.ifft();
	kernel_Phi4_Phi6<<<grid, block>>>(N, Grid.get_t_ptr(), Grid.get_q_ptr(), Grid.get_lambda(), Grid.get_g());
	cudaDeviceSynchronize();
	Grid.doFFT_t2T();
}