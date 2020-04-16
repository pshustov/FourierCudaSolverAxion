#include "stdafx.h"

int inWhichInterval(const unsigned int N, const unsigned leftPowN, const double a, const double* b)
{
	if (a >= b[N] || a < b[0]) {
		return -1;
	}

	int k = N;
	int l = N;

	for (unsigned int i = 0; i < leftPowN; i++)
	{
		l >>= 1;
		if (N & (1 << i))
		{
			if (a < b[k - l])
			{
				if (a < b[k - l - 1])
				{
					k -= l + 1;
				}
				else
				{
					return k - l - 1;
				}
			}
		}
		else
		{
			if (a < b[k - l])
			{
				k -= l;
			}
		}
	}
	throw;
}

Distribution::Distribution(cudaGrid_3D & Grid)
{
	f.set_size_erase(Nf);
	fDistrOut.open("fDistr.txt");

	int N1 = (int) Grid.get_N1();
	int N2 = (int) Grid.get_N2();
	int N3 = (int) Grid.get_N3();
	int N3red = (int) Grid.get_N3red();

	rhoK.set_size_erase(N1, N2, N3red);
	omega.set_size_erase(N1, N2, N3red);

	double L1 = Grid.get_L1();
	double L2 = Grid.get_L2();
	double L3 = Grid.get_L3();

	if (N1 == N2 && N1 == N3) { N = N1; }
	else { throw; }

	if (abs(L1 - L2) / L1 < 1e-10 && abs(L1 - L3) / L1 < 1e-10) { kappa = 2 * M_PI / L1; }
	else { throw; }

	double d = double(N) / 2.0 / double(Nf);
	for (int i = 0; i < Nf + 1; i++) {
		boundariesSqr[i] = d * i * d * i;
	}

	fDistrOut << Nf << "\t";
	for (int i = 0; i < Nf - 1; i++) {
		fDistrOut << kappa * d * (i + 0.5) << "\t";
	}
	fDistrOut << kappa * d * (Nf - 0.5) << std::endl;
}

void Distribution::setDistributionFunction(const complex* rhoKcuda, const double* omegaCuda)
{
	rhoK.copyFromCudaPtr(rhoKcuda);
	omega.copyFromCudaPtr(omegaCuda);


	for (int i = 0; i < f.get_N(); i++)
	{
		f(i) = 0;
	}

	int ind = 0;
	double rsqr, r1sqr, r2sqr, r3sqr;
	for (int i = 0; i < N; i++)
	{
		(i <= N / 2) ? r1sqr = i * i : r1sqr = (i - N) * (i - N);
		for (int j = 0; j < N; j++)
		{
			(j <= N / 2) ? r2sqr = j * j : r2sqr = (j - N) * (j - N);
			for (int k = 0; k < N; k++)
			{
				if (k <= N / 2)
				{
					r3sqr = k * k;
					rsqr = r1sqr + r2sqr + r3sqr;

					ind = inWhichInterval(Nf, powNf, rsqr, boundariesSqr);
					if (ind > 0) {
						if (i != 0 || j != 0 || k != 0)
						{
							f(ind) += rhoK(i, j, k) / omega(i, j, k);
						}
					}
				}
				else
				{
					r3sqr = (k - N) * (k - N);
					rsqr = r1sqr + r2sqr + r3sqr;

					ind = inWhichInterval(Nf, powNf, rsqr, boundariesSqr);
					if (ind > 0) {
						if (i == 0 && j == 0)
						{
							f(ind) += rhoK(0, 0, N - k).get_conj() / omega(0, 0, N - k);
						}
						else if (i == 0)
						{
							f(ind) += rhoK(0, N - j, N - k).get_conj() / omega(0, N - j, N - k);
						}
						else if (j == 0)
						{
							f(ind) += rhoK(N - i, 0, N - k).get_conj() / omega(N - i, 0, N - k);
						}
						else
						{
							f(ind) += rhoK(N - i, N - j, N - k).get_conj() / omega(N - i, N - j, N - k);
						}
					}
				}


			}
		}
	}

	printDistr();

}

