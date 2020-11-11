#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "cudaGrid.h"

cudaGrid_3D::cudaGrid_3D(const std::string& filename)
{
	int priority_high, priority_low;
	checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
	checkCudaErrors(cudaStreamCreateWithPriority(&mainStream, cudaStreamNonBlocking, priority_high));
	checkCudaErrors(cudaStreamCreateWithPriority(&printStream, cudaStreamNonBlocking, priority_low));

	N1 = 0; N2 = 0; N3 = 0; N3red = 0;
	L1 = 0; L2 = 0; L3 = 0;
	current_time = 0;
	energy = 0;	energyPrev = 0;

	///must be identical with save()
	N1buf = 32;	N2buf = 32;	N3buf = 32;

	// loading
	load(filename);

	// set x and k
	set_xk();

	//set other parameters

	int n_fft[3];
	n_fft[0] = (int)N1;
	n_fft[1] = (int)N2;
	n_fft[2] = (int)N3;
	cufft.reset(3, n_fft, getVolume(), 1, mainStream);

	fft();
	
	std::cout << "First FFT done\n";

	isIFFTsyncQ = true;
	isIFFTsyncP = true;
	isEnergyCalculateted = false;
	isQPsqrCalculated = false;
}

cudaGrid_3D::~cudaGrid_3D()
{
}

void cudaGrid_3D::fft()
{
	cufft.forward(q, Q);
	cufft.forward(p, P);
}

void cudaGrid_3D::ifft()
{
	ifftQ();
	ifftP();
}

void cudaGrid_3D::ifftQ()
{
	if (!isIFFTsyncQ)
	{
		cufft.inverce(Q, q);
		isIFFTsyncQ = true;
	}
}

void cudaGrid_3D::ifftP()
{
	if (!isIFFTsyncP)
	{
		cufft.inverce(P, p);
		isIFFTsyncP = true;
	}
}

void cudaGrid_3D::set_sizes()
{
	if (N3 % 2 != 0) { throw; }

	x1.set(N1);
	x2.set(N2);
	x3.set(N3);
	k1.set(N1, N2, N3red);
	k2.set(N2, N2, N3red);
	k3.set(N3, N2, N3red);

	k_sqr.set(N1, N2, N3red);

	q.set(N1, N2, N3);
	p.set(N1, N2, N3);
	t.set(N1, N2, N3);
	qpSqr.set(N1, N2, N3);
	Q.set(N1, N2, N3red);
	P.set(N1, N2, N3red);
	T.set(N1, N2, N3red);

	buferOutHost.set(N1buf, N2buf, N3buf);
	buferOutDev.set(N1buf, N2buf, N3buf);
	if (N1 / N1buf > 1 || N2 / N2buf > 1 || N3 / N3buf > 1) { isBuferising = true; }
	else { isBuferising = false; }
}

void cudaGrid_3D::set_xk()
{
	real kappa1 = 2 * Ma_PI / L1;
	real kappa2 = 2 * Ma_PI / L2;
	real kappa3 = 2 * Ma_PI / L3;

	RVector temp1(N1), temp2(N2), temp3(N3);
	RVector3 temp31(N1, N2, N3red), temp32(N1, N2, N3red), temp33(N1, N2, N3red);

	/// set x1 x2 x3
	for (size_t i = 0; i < N1; i++) { temp1(i) = L1 * i / N1; }
	x1 = temp1;

	for (size_t i = 0; i < N2; i++) { temp2(i) = L2 * i / N2; }
	x2 = temp2;

	for (size_t i = 0; i < N3; i++) { temp3(i) = L3 * i / N3; }
	x3 = temp3;

	/// set k1 k2 k3
	for (size_t i = 0; i < N1; i++) { temp1(i) = (i < N1 / 2) ? (kappa1 * i) : (i > N1 / 2 ? (kappa1 * i - kappa1 * N1) : 0); }
	for (size_t i = 0; i < N2; i++) { temp2(i) = (i < N2 / 2) ? (kappa2 * i) : (i > N2 / 2 ? (kappa2 * i - kappa2 * N2) : 0); }
	for (size_t i = 0; i < N3; i++) { temp3(i) = (i < N3 / 2) ? (kappa3 * i) : (i > N3 / 2 ? (kappa3 * i - kappa3 * N3) : 0); }

	for (size_t i = 0; i < N1; i++)
	{
		for (size_t j = 0; j < N2; j++)
		{
			for (size_t k = 0; k < N3red; k++)
			{
				temp31(i, j, k) = temp1(i);
				temp32(i, j, k) = temp2(j);
				temp33(i, j, k) = temp3(k);
			}
		}
	}
	k1 = temp31;
	k2 = temp32;
	k3 = temp33;

	///set k_sqr
	for (size_t i = 0; i < N1; i++) { temp1(i) = (i <= N1 / 2) ? ((kappa1 * i)*(kappa1 * i)) : ((kappa1 * i - kappa1 * N1) * (kappa1 * i - kappa1 * N1)); }
	for (size_t i = 0; i < N2; i++) { temp2(i) = (i <= N2 / 2) ? ((kappa2 * i)*(kappa2 * i)) : ((kappa2 * i - kappa2 * N2) * (kappa2 * i - kappa2 * N2)); }
	for (size_t i = 0; i < N3; i++) { temp3(i) = (i <= N3 / 2) ? ((kappa3 * i)*(kappa3 * i)) : ((kappa3 * i - kappa3 * N3) * (kappa3 * i - kappa3 * N3)); }

	for (size_t i = 0; i < N1; i++)
	{
		for (size_t j = 0; j < N2; j++)
		{
			for (size_t k = 0; k < N3red; k++)
			{
				temp31(i, j, k) = temp1(i) + temp2(j) + temp3(k);
			}
		}
	}
	k_sqr = temp31;
}



__global__ void kernelEnergyQuad(cudaRVector3Dev kSqr, cudaCVector3Dev Q, cudaCVector3Dev P, cudaCVector3Dev T)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;

	size_t N1 = kSqr.getN1(), N2 = kSqr.getN2(), N3 = kSqr.getN3();

	size_t ind = (i * N2 + j) * N3 + k;

	if (i < N1 && j < N2 && k < N3)
	{
		if (k == 0) {
			T(ind) = (P(ind).absSqr() + (1 + kSqr(ind)) * Q(ind).absSqr()) / 2.0;
		}
		else {
			T(ind) = (P(ind).absSqr() + (1 + kSqr(ind)) * Q(ind).absSqr());

		}
	}
}

__global__ void kernelEnergyNonLin(real lam, real g, real V, cudaRVector3Dev q, cudaRVector3Dev t)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	real f = q(i);
	if (i < q.getSize())
	{
		t(i) = (lam / 4.0 + g / 6.0 * f * f) * f * f * f * f;
	}
}

void cudaGrid_3D::calculateEnergy()
{
	if (!isEnergyCalculateted)
	{
		energyPrev = energy;

		int Bx = 16, By = 8, Bz = 1;
		dim3 block(Bx, By, Bz);
		dim3 grid((static_cast<unsigned int>(N1) + Bx - 1) / Bx, (static_cast<unsigned int>(N2) + By - 1) / By, (static_cast<unsigned int>(N3red) + Bz - 1) / Bz);
		kernelEnergyQuad << <grid, block, 0, mainStream >> > (k_sqr, Q, P, T);
		energy = T.getSum(mainStream).get_real() / getVolume();

		ifftQ();

		real V = getVolume();

		block = dim3(BLOCK_SIZE);
		grid = dim3((static_cast<unsigned int>(getSize()) + BLOCK_SIZE - 1) / BLOCK_SIZE);
		kernelEnergyNonLin << <grid, block, 0, mainStream >> > (lambda, g, getVolume(), q, t);
		energy += t.getSum(mainStream) * getVolume() / getSize();

		isEnergyCalculateted = true;
	}
}

real cudaGrid_3D::getEnergy() {
	calculateEnergy();
	return energy;
}

real cudaGrid_3D::getEnergyPrev() {
	calculateEnergy();
	return energyPrev;
}


__global__ void kernelQPsqr(cudaRVector3Dev q, cudaRVector3Dev p, cudaRVector3Dev qpSqr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < q.getSize()) { qpSqr(i) = 0.5 * (q(i) * q(i) + p(i) * p(i)); }
}

void cudaGrid_3D::calculateQPsqr()
{
	if (!isQPsqrCalculated)
	{
		dim3 block(BLOCK_SIZE);
		dim3 grid((static_cast<unsigned int>(q.getSize()) + BLOCK_SIZE - 1) / BLOCK_SIZE);

		ifft();
		kernelQPsqr<<< grid, block, 0, mainStream >>>(q, p, qpSqr);
		cudaStreamSynchronize(mainStream);
		isQPsqrCalculated = true;
	}
}

real cudaGrid_3D::getMaxValQsqr()
{
	calculateQPsqr();
	return qpSqr.getMax(printStream);
}


__global__ void kernelSyncBuf(real* A, real* A0)
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

	extern __shared__ real B[];
	B[indB] = A0[indA0];
	__syncthreads();


	int numOfElem = N1 * N2 * N3;
	int step = 1;
	while (numOfElem > 1)
	{
		if (indB % (2 * step) == 0)
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


void cudaGrid_3D::printingVTK(std::ofstream& outVTK)
{
	calculateQPsqr();
	if (isBuferising)
	{
		int factor1 = static_cast<int>((N1 + N1buf - 1) / N1buf), factor2 = static_cast<int>((N2 + N2buf - 1) / N2buf), factor3 = static_cast<int>((N3 + N3buf - 1) / N3buf);

		dim3 grid(static_cast<unsigned int>(N1buf), static_cast<unsigned int>(N2buf), static_cast<unsigned int>(N3buf));
		dim3 block(factor1, factor2, factor3);
		
		kernelSyncBuf<<< grid, block, factor1 * factor2 * factor3 * sizeof(real), printStream >>>(buferOutDev.getArray(), qpSqr.getArray());
		cudaStreamSynchronize(printStream);
		buferOutHost = buferOutDev;
	}
	else
	{
		buferOutHost = qpSqr;
	}

	for (size_t i = 0; i < buferOutHost.getSize(); i++) {
		outVTK << buferOutHost(i) << '\n';
	}
	outVTK << std::flush;
}

void cudaGrid_3D::save(std::ofstream& fileSave)
{
	fileSave << getN1() << "\n" << getN2() << "\n" << getN3() << "\n";
	fileSave << getL1() << "\n" << getL2() << "\n" << getL3() << "\n";
	fileSave << f0 << "\n";
	fileSave << sigma << "\n";
	fileSave << p0 << "\n";

	ifft();

	RVector3 RHost;
	RHost = q;
	for (size_t i = 0; i < RHost.getSize(); i++) {
		fileSave << RHost(i) << '\n';
	}

	RHost = p;
	for (size_t i = 0; i < RHost.getSize(); i++) {
		fileSave << RHost(i) << '\n';
	}

	fileSave << current_time;
}

void cudaGrid_3D::load(const std::string& filename)
{
	std::ifstream in(filename);

	in >> N1;	in >> N2; 	in >> N3;
	N3red = N3 / 2 + 1;
	in >> L1;	in >> L2;	in >> L3;
	in >> f0;
	in >> sigma;
	in >> p0;

	//check N1 N2 N3
	if (N1 / N_MIN == 0 || N2 / N_MIN == 0 || N3 / N_MIN == 0) { throw; }

	//malloc for all varibles 
	set_sizes();

	// set q p
	RVector3 RHost(N1, N2, N3);
	for (size_t i = 0; i < N1 * N2 * N3; i++) {
		in >> RHost(i);
	}
	q = RHost;

	for (size_t i = 0; i < N1 * N2 * N3; i++) {
		in >> RHost(i);
	}
	p = RHost;

	if (!in.eof()) {
		in >> current_time;
	}
	else {
		current_time = 0;
	}

	in.close();
	std::cout << "Loading have been done\n";

}