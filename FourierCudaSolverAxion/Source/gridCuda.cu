#include "stdafx.h"

double reductionSum(int size, double *inData);

cudaGrid_3D::cudaGrid_3D(const std::string filename)
{
	current_time = 0;

	std::ifstream in(filename);

	in >> N1;	in >> N2; 	in >> N3;
	in >> L1;	in >> L2;	in >> L3;

	//check N1 N2 N3
	if (N1 % N_MIN != 0 || N2 % N_MIN != 0 || N3 % N_MIN != 0) { throw; }

	//malloc for all varibles 
	set_sizes();

	// set q p
	for (size_t i = 0; i < N1*N2*N3; i++) {
		in >> RHost(i);
	}
	q = RHost;

	for (size_t i = 0; i < N1*N2*N3; i++) {
		in >> RHost(i);
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
	cufft.reset(3, n_fft, get_volume());

	//fft
	fft();

	std::cout << "First FFT have been done\n";

	isIFFTsync = true;
	isRhoCalculateted = false;
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
	if (!isIFFTsync)
	{
		cufft.inverce(Q, q);
		cufft.inverce(P, p);
	}
	isIFFTsync = true;
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
	Q.set(N1, N2, N3red);
	P.set(N1, N2, N3red);
	T.set(N1, N2, N3red);

	rho.set(N1, N2, N3);

	RHost.set_size_erase(N1, N2, N3);
	CHost.set_size_erase(N1, N2, N3red);
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
