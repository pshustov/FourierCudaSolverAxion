#include "stdafx.h"

cudaGrid_3D::cudaGrid_3D(const std::string filename)
{
	mainStream = cudaStreamDefault;
	//cudaStreamCreate(&mainStream);
	cudaStreamCreate(&printStream);

	current_time = 0;

	std::ifstream in(filename);

	in >> N1;	in >> N2; 	in >> N3;
	in >> L1;	in >> L2;	in >> L3;

	//check N1 N2 N3
	if (N1 / N_MIN == 0 || N2 / N_MIN == 0 || N3 / N_MIN == 0) { throw; }

	//malloc for all varibles 
	set_sizes();

	// set q p
	RVector3 RHost(N1, N2, N3);
	for (size_t i = 0; i < N1*N2*N3; i++) {
		//in >> RHost(i);
		RHost(i) = 10;
	}
	q = RHost;

	for (size_t i = 0; i < N1*N2*N3; i++) {
		//in >> RHost(i);
		RHost(i) = 10;
	}
	p = RHost;

	if (!in.eof())
	{
		in >> current_time;
	}
	in.close();
	std::cout << "Loading have been done\n";

	// set x and k
	set_xk();

	//set other parameters

	int n_fft[3];
	n_fft[0] = (int)N1;
	n_fft[1] = (int)N2;
	n_fft[2] = (int)N3;
	cufft.reset(3, n_fft, getVolume(), 1, mainStream);

	//fft
	//fft
	fft();

	//ifft();
	//RHost = q;
	//std::ofstream ftest("test.txt");
	//for (int i = 0; i < size(); i++)
	//{
	//	ftest << RHost(i) << "\n";
	//}

	std::cout << "First FFT have been done\n";

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
	N3red = N3 / 2 + 1;

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
	double kappa1 = 2 * Ma_PI / L1;
	double kappa2 = 2 * Ma_PI / L2;
	double kappa3 = 2 * Ma_PI / L3;

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

__global__ void kernelEnergyNonLin(double lam, double g, double V, cudaRVector3Dev q, cudaRVector3Dev t)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	double f = q(i);
	if (i < q.size())
	{
		t(i) = (lam / 4.0 + g / 6.0 * f * f) * f * f * f * f;
	}
}

double cudaGrid_3D::getEnergy()
{
	if (!isEnergyCalculateted)
	{
		int Bx = 16, By = 8, Bz = 1;
		dim3 block(Bx, By, Bz);
		dim3 grid((N1 + Bx - 1) / Bx, (N2 + By - 1) / By, (N3red + Bz - 1) / Bz);
		kernelEnergyQuad<<<grid, block, 0, mainStream>>>(k_sqr, Q, P, T);
		cudaStreamSynchronize(mainStream);
		energy = T.getSum(mainStream).real() / getVolume();
		
		ifftQ();

		double V = getVolume();

		block = dim3(BLOCK_SIZE);
		grid = dim3((size() + BLOCK_SIZE - 1) / BLOCK_SIZE);
		kernelEnergyNonLin<<<grid, block, 0, mainStream>>>(lambda, g, getVolume(), q, t);
		cudaStreamSynchronize(mainStream);
		energy += t.getSum(mainStream) * getVolume() / size();

		isEnergyCalculateted = true;
	}
	return energy;
}


__global__ void kernelQPsqr(cudaRVector3Dev q, cudaRVector3Dev p, cudaRVector3Dev qpSqr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < q.size()) { qpSqr(i) = 0.5 * (q(i) * q(i) + p(i) * p(i)); }
}

void cudaGrid_3D::calculateQPsqr()
{
	if (!isQPsqrCalculated)
	{
		dim3 block(BLOCK_SIZE);
		dim3 grid((q.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);	

		ifft();
		kernelQPsqr<<< grid, block, 0, mainStream >>>(q, p, qpSqr);
		cudaStreamSynchronize(mainStream);
		isQPsqrCalculated = true;
	}
}

double cudaGrid_3D::getMaxValQsqr()
{
	calculateQPsqr();
	return qpSqr.getMax(printStream);
}


void cudaGrid_3D::printingVTK(std::ofstream& outVTK)
{
	calculateQPsqr();
	if (isBuferising)
	{
		int factor1 = (N1 + N1buf - 1) / N1buf, factor2 = (N2 + N2buf - 1) / N2buf, factor3 = (N3 + N3buf - 1) / N3buf;

		dim3 grid(N1buf, N2buf, N3buf);
		dim3 block(factor1, factor2, factor3);
		
		kernelSyncBuf<<< grid, block, factor1 * factor2 * factor3 * sizeof(double), printStream >>>(buferOutDev.getArray(), qpSqr.getArray());
		cudaStreamSynchronize(printStream);
		buferOutHost = buferOutDev;
	}
	else
	{
		buferOutHost = qpSqr;
	}

	for (size_t i = 0; i < buferOutHost.size(); i++) {
		outVTK << buferOutHost(i) << '\n';
	}
	outVTK << std::flush;
}