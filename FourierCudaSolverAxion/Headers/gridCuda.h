#pragma once

#include "stdafx.h"

class cudaGrid_3D
{
public:
	cudaGrid_3D(const std::string filename);
	~cudaGrid_3D();

	void fft();
	void ifft();
	void ifftQ();
	void ifftP();

	void save(std::ofstream & fileSave)
	{		
		fileSave << getN1() << "\n" << getN2() << "\n" << getN3() << "\n";
		fileSave << getL1() << "\n" << getL2() << "\n" << getL3() << "\n";

		ifft();

		RVector3 RHost;
		RHost = q;
		for (size_t i = 0; i < RHost.size(); i++) {
			fileSave << RHost(i) << '\n';
		}
		
		RHost = p;
		for (size_t i = 0; i < RHost.size(); i++) {
			fileSave << RHost(i) << '\n';
		}
	}

	void load(const std::string filename)
	{
		std::ifstream inFile(filename);
	}

	/// Gets
	size_t size() const { return N1*N2*N3; }
	size_t getN1() const { return N1; }
	size_t getN2() const { return N2; }
	size_t getN3() const { return N3; }
	size_t getN3red() const { return N3red; }
	size_t getN1buf() const { return N1buf; }
	size_t getN2buf() const { return N2buf; }
	size_t getN3buf() const { return N3buf; }

	double getL1() const { return L1; }
	double getL2() const { return L2; }
	double getL3() const { return L3; }
	double getVolume() const { return L1 * L2 * L3; }

	cudaRVector get_x1() const { return x1; }
	cudaRVector get_x2() const { return x2; }
	cudaRVector get_x3() const { return x3; }
	cudaRVector3 get_k1() const { return k1; }
	cudaRVector3 get_k2() const { return k2; }
	cudaRVector3 get_k3() const { return k3; }
	cudaRVector3& get_k_sqr() { return k_sqr; }

	cudaRVector3& get_q() { return q; }
	cudaRVector3& get_p() { return p; }
	cudaRVector3& get_t() { return t; }
	cudaCVector3& get_Q() { return Q; }
	cudaCVector3& get_P() { return P; }
	cudaCVector3& get_T() { return T; }

	double get_time() const { return current_time; }
	double get_lambda() const { return lambda; }
	double get_g() const { return g; }
	cudaStream_t get_mainStream() const { return mainStream; }

	/// FFT and IFFT
	void doFFTforward(cudaCVector3 &f, cudaCVector3 &F) { cufft.forward(f, F); }
	void doFFTforward(cudaRVector3 &f, cudaCVector3 &F) { cufft.forward(f, F); }
	void doFFTinverce(cudaCVector3 &F, cudaCVector3 &f) { cufft.inverce(F, f); }
	void doFFTinverce(cudaCVector3 &F, cudaRVector3 &f) { cufft.inverce(F, f); }


	/// Sets 
	void set_lambda(const double _lambda) { lambda = _lambda; }
	void set_g(const double _g) { g = _g; }
	void setSmthChanged() { 
		isEnergyCalculateted = false; 
		isIFFTsyncQ = false;
		isIFFTsyncP = false;
		isQPsqrCalculated = false;
	}

	/// Other methods 
	double get_dt(const double precision) const
	{
		return precision;
	}
	void set_sizes();
	void set_xk();
	void timestep(double dt) { setSmthChanged();  current_time += dt; }
	double getEnergy();

	void calculateQPsqr();
	double getMaxValQsqr();
	
	void printingVTK(std::ofstream& outVTK);

private:
	size_t N1, N2, N3, N3red;
	double L1, L2, L3;

	cudaRVector x1, x2, x3;
	cudaRVector3 k1, k2, k3;
	cudaRVector3 k_sqr;
	cudaRVector3 q, p, t, qpSqr;
	cudaCVector3 Q, P, T;

	RVector3 buferOutHost;
	cudaRVector3 buferOutDev;
	size_t N1buf = 32, N2buf = 32, N3buf = 32;
	bool isBuferising;

	cuFFT cufft;

	double lambda, g;
	double current_time;
	bool isIFFTsyncQ, isIFFTsyncP, isEnergyCalculateted, isQPsqrCalculated;

	double energy;

	cudaStream_t mainStream, printStream;
};

