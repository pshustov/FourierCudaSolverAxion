// fourierEquationSolving.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


__global__ void kernalStepSymplectic41(const int N, const double dt,
	double *k_sqr, complex *Q, complex *P, complex *T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		P[i] -= 0.67560359597982881702384390448573 * (k_sqr[i] * Q[i] + Q[i] + T[i]) * dt;
		Q[i] += 1.3512071919596576340476878089715 * P[i] * dt;
	}
}


__global__ void kernalStepSymplectic42(const int N, const double dt,
	double *k_sqr, complex *Q, complex *P, complex *T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		P[i] -= -0.17560359597982881702384390448573 * (k_sqr[i] * Q[i] + Q[i] + T[i]) * dt;
		Q[i] += -1.702414383919315268095375617943 * P[i] * dt;
	}
}


__global__ void kernalStepSymplectic43(const int N, const double dt,
	double *k_sqr, complex *Q, complex *P, complex *T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		P[i] -= -0.17560359597982881702384390448573 * (k_sqr[i] * Q[i] + Q[i] + T[i]) * dt;
		Q[i] += 1.3512071919596576340476878089715 * P[i] * dt;
	}
}


__global__ void kernalStepSymplectic44(const int N, const double dt,
	double *k_sqr, complex *Q, complex *P, complex *T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		P[i] -= 0.67560359597982881702384390448573 * (k_sqr[i] * Q[i] + Q[i] + T[i]) * dt;
	}
}


__global__ void kernel_Phi4_Phi6(const int N, double *t, double *q, const double lambda, const double g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		t[i] = q[i] * q[i] * q[i] * (lambda + g * q[i] * q[i]);
	}
}


__global__ void kernelAddMullSqr(const int N, double* S, double* A, double m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		S[i] += m * A[i] * A[i];
	}
}

__global__ void kernelCulcRhoReal(const int N, double *rho, double *q, double *p, const double lambda, const double g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		double qi = q[i];
		double pi = p[i];

		rho[i] = 0.5 * qi * qi;
		rho[i] += 0.5 * pi * pi;
		rho[i] += (lambda / 4.0) * qi * qi * qi * qi;
		rho[i] += (g / 6.0)  * qi * qi * qi * qi * qi * qi;
	}
}

__global__ void kernelDer(const int N, complex* T, double *k, complex *Q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		T[i] = complex(0, 1) * k[i] * Q[i];
	}
}


__global__ void kernelSyncBuf(double *A, double *A0)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int k = threadIdx.z;
	const int N1 = blockDim.x;
	const int N2 = blockDim.y;
	const int N3 = blockDim.z;

	const int iB = blockIdx.x;
	const int jB = blockIdx.y;
	const int kB = blockIdx.z;
	//const int N1B = gridDim.x;	//just never used
	const int N2B = gridDim.y;
	const int N3B = gridDim.z;
	
	const int iG = i + iB * N1;
	const int jG = j + jB * N2;
	const int kG = k + kB * N3;
	//const int N1G = N1 * N1B;		//just never used
	const int N2G = N2 * N2B;
	const int N3G = N3 * N3B;

	const int indB = k + N3 * (j + N2 * i);
	const int indA = kB + N3B * (jB + N2B * iB);
	const int indA0 = kG + N3G * (jG + N2G * iG);

	extern __shared__ double B[];
	B[indB] = A0[indA0];
	__syncthreads();


	int numOfElem = N1 * N2 * N3;		
	int step = 1;
	while (numOfElem > 1)
	{
		if (indB % (2*step) == 0)
		{
			B[indB] = B[indB] + B[indB + step];
		}
		__syncthreads();

		numOfElem /= 2;
		step *= 2;

	}

	if (indB == 0)
	{
		A[indA] = B[0] / (N1 * N2 * N3);
	}

}


__global__ void kernelGetOmega(const int N, double *omega, double *kSqr, const double sigma2, const double sigma4, const double lambda, const double g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		omega[i] = sqrt(1 + kSqr[i] + 3 * lambda * sigma2 + 15 * g * sigma4);
	}
}


__global__ void kernelSetRhoK(complex *T, double m, double *k_sqr, complex *Q, complex *P)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	T[i] = m * (P[i] * P[i].get_conj() + (1 + k_sqr[i]) * Q[i] * Q[i].get_conj());
}

__global__ void kernelAddRhoK(double m, complex *Q, complex *T)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	T[i] = m * Q[i] * T[i].get_conj();
}

__global__ void kernelGetPhi2(const int N, double *T, double *q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		T[i] = q[i] * q[i];
	}
}

__global__ void kernelGetPhi3(const int N, double *T, double *q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		T[i] = q[i] * q[i] * q[i];
	}
}

__global__ void kernelGetPhi5(const int N, double *T, double *q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		T[i] = q[i] * q[i] * q[i] * q[i] * q[i];
	}
}