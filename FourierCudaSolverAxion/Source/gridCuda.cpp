#include "stdafx.h"

//double reductionSigma2(int size, double *inData);
//double reductionSigma4(int size, double *inData);
double reductionSum(int size, double *inData);

int getPow2(unsigned int N)
{
	int p = 0;
	if (N&(N - 1) != 0) {
		throw;
	}
	while (N >>= 1) {
		p++;
	}
	return p;
}

cudaGrid_3D::cudaGrid_3D(const std::string filename)
{
	current_time = 0;

	std::ifstream in(filename);

	in >> N1;	in >> N2; 	in >> N3;
	in >> L1;	in >> L2;	in >> L3;


	//check N1 N2 N3
	if (N1 % N_MIN != 0 || N2 % N_MIN != 0 || N3 % N_MIN != 0) { throw; }

	//set all others Ns 
	int N_current_min = (int) (N1 < N2 ? (N1 < N3 ? N1 : N3) : (N2 < N3 ? N2 : N3));
	factor = N_current_min / N_MIN;

	if (factor > 2) {
		N1_print = N1 / factor;
		N2_print = N2 / factor;
		N3_print = N3 / factor;
	}
	else
	{
		N1_print = N1;
		N2_print = N2;
		N3_print = N3;
	}

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
	cufft.reset(3, n_fft);

	//fft
	fft();

	std::cout << "First FFT have been done\n";

	isIFFTsync = true;
	isRhoCalculateted = false;
	isRhoKCalculateted = false;
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


void cudaGrid_3D::printingVTK(std::ofstream & outVTK) const
{
	int M = (int)RHost.size();
	for (size_t i = 0; i < M; i++) {
		outVTK << RHost(i) << '\n';
	}
	outVTK << std::flush;
}


void cudaGrid_3D::hostSynchronize_q()
{
	ifft();
	if (factor > 2)
	{
		dim3 grid((unsigned int) N1_print, (unsigned int) N2_print, (unsigned int) N3_print);
		dim3 block(factor, factor, factor);

		kernelSyncBuf << <grid, block, factor*factor*factor * sizeof(double) >> > (get_buf_ptr(), get_q_ptr());
		cudaDeviceSynchronize();
		RHost = bufPrint;
	}
	else
	{
		RHost = q;
	}
}


void cudaGrid_3D::hostSynchronize_p()
{
	ifft();
	if (factor > 2)
	{
		dim3 grid((unsigned int)N1_print, (unsigned int)N2_print, (unsigned int)N3_print);
		dim3 block(factor, factor, factor);

		kernelSyncBuf << <grid, block, factor*factor*factor * sizeof(double) >> > (get_buf_ptr(), get_p_ptr());
		cudaDeviceSynchronize();
		RHost = bufPrint;
	}
	else
	{
		RHost = p;
	}
}


void cudaGrid_3D::hostSynchronize_rho()
{
	ifft(); 
	if (factor > 2)
	{
		dim3 grid((unsigned int)N1_print, (unsigned int)N2_print, (unsigned int)N3_print);
		dim3 block(factor, factor, factor);

		kernelSyncBuf << <grid, block, factor*factor*factor * sizeof(double) >> > (get_buf_ptr(), get_rho_ptr());
		cudaDeviceSynchronize();
		RHost = bufPrint;
	}
	else
	{
		RHost = rho;
	}
}


void cudaGrid_3D::set_sizes()
{
	if (N3 % 2 != 0) { throw; }
	N3red = N3 / 2 + 1;

	x1.set_size_erase(N1);
	x2.set_size_erase(N2);
	x3.set_size_erase(N3);
	k1.set_size_erase(N1, N2, N3red);
	k2.set_size_erase(N2, N2, N3red);
	k3.set_size_erase(N3, N2, N3red);

	k_sqr.set_size_erase(N1, N2, N3red);

	q.set_size_erase(N1, N2, N3);
	p.set_size_erase(N1, N2, N3);
	t.set_size_erase(N1, N2, N3);
	Q.set_size_erase(N1, N2, N3red);
	P.set_size_erase(N1, N2, N3red);
	T.set_size_erase(N1, N2, N3red);

	rho.set_size_erase(N1, N2, N3);
	rhoK.set_size_erase(N1, N2, N3red);
	omega.set_size_erase(N1, N2, N3red);

	RHost.set_size_erase(N1, N2, N3);
	CHost.set_size_erase(N1, N2, N3red);

	bufPrint.set_size_erase(N1_print, N2_print, N3_print);
}


void cudaGrid_3D::set_xk()
{
	double kappa1 = 2 * M_PI / L1;
	double kappa2 = 2 * M_PI / L2;
	double kappa3 = 2 * M_PI / L3;

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


void cudaGrid_3D::calculateRho()
{
	if (!isRhoCalculateted)
	{

		ifft();

		const int N = (int)(N1 * N2 * N3);
		const int Nred = (int)(N1 * N2 * N3red);

		dim3 block(BLOCK_SIZE);
		dim3 grid((unsigned int)ceil((double)N / (double)BLOCK_SIZE));
		dim3 gridRed((unsigned int)ceil((double)Nred / (double)BLOCK_SIZE));

		kernelCulcRhoReal << <grid, block >> > (N, get_rho_ptr(), get_q_ptr(), get_p_ptr(), lambda, g);
		cudaDeviceSynchronize();

		///AddDir1
		kernelDer << <gridRed, block >> > (Nred, get_T_ptr(), get_k1_ptr(), get_Q_ptr());
		cudaDeviceSynchronize();
		doFFT_T2t();
		kernelAddMullSqr << <grid, block >> > (N, get_rho_ptr(), get_t_ptr(), 0.5);

		///AddDir2
		kernelDer << <grid, block >> > (Nred, get_T_ptr(), get_k2_ptr(), get_Q_ptr());
		cudaDeviceSynchronize();
		doFFT_T2t();
		kernelAddMullSqr << <grid, block >> > (N, get_rho_ptr(), get_t_ptr(), 0.5);

		///AddDir3
		kernelDer << <grid, block >> > (Nred, get_T_ptr(), get_k3_ptr(), get_Q_ptr());
		cudaDeviceSynchronize();
		doFFT_T2t();
		kernelAddMullSqr << <grid, block >> > (N, get_rho_ptr(), get_t_ptr(), 0.5);

		isRhoCalculateted = true;
	}
}


void cudaGrid_3D::calculateRhoK()
{
	if (!isRhoKCalculateted)
	{
		const int N = (int)(N1 * N2 * N3);
		const int Nred = (int)(N1 * N2 * N3red);

		dim3 block(BLOCK_SIZE);
		dim3 grid((unsigned int)ceil((double)N / (double)BLOCK_SIZE));
		dim3 gridRed((unsigned int)ceil((double)Nred / (double)BLOCK_SIZE));
		
		double m = 1 / (2. * 2. * M_PI * 2. * M_PI * 2. * M_PI);
		kernelSetRhoK<<<gridRed, block>>>(rhoK.get_Array(), m, k_sqr.get_Array(), Q.get_Array(), P.get_Array());
		cudaDeviceSynchronize();
		
		ifft();
		kernelGetPhi3<<<grid, block>>>(N, t.get_Array(), q.get_Array());
		cudaDeviceSynchronize();
		doFFTforward(t, T);
		m = lambda / (4. * 2. * M_PI * 2. * M_PI * 2. * M_PI);
		kernelAddRhoK<<<gridRed, block>>>(m, Q.get_Array(), T.get_Array());
		cudaDeviceSynchronize();

		kernelGetPhi5<<<grid, block>>>(N, t.get_Array(), q.get_Array());
		cudaDeviceSynchronize();
		doFFTforward(t, T);
		m = g / (6. * 2. * M_PI * 2. * M_PI * 2. * M_PI);
		kernelAddRhoK<<<gridRed, block>>>(m, Q.get_Array(), T.get_Array());
		cudaDeviceSynchronize();

		isRhoKCalculateted = true;
	}
}


void cudaGrid_3D::calculateOmega()
{
	double sigma2, sigma4;

	const int N = (int)(N1 * N2 * N3);
	const int Nred = (int)(N1 * N2 * N3red);

	dim3 block(BLOCK_SIZE);
	dim3 grid((unsigned int)ceil((double)N / (double)BLOCK_SIZE));
	dim3 gridRed((unsigned int)ceil((double)Nred / (double)BLOCK_SIZE));

	ifft();
	kernelGetPhi2<<<grid, block>>>(N, t.get_Array(), q.get_Array());

	sigma2 = reductionSum(N, q.get_Array()) / N;
	sigma4 = sigma2 * sigma2;
	
	cudaDeviceSynchronize();
	kernelGetOmega<<<gridRed, block>>>(Nred, omega.get_Array(), k_sqr.get_Array(), sigma2, sigma4, lambda, g);
	cudaDeviceSynchronize();
}