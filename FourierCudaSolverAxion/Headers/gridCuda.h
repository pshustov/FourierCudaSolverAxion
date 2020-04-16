#pragma once

#include "stdafx.h"

class cudaGrid_3D
{
public:
	cudaGrid_3D(const std::string filename);
	~cudaGrid_3D();

	void fft();
	void ifft();

	void printingVTK(std::ofstream & outVTK) const;

	void hostSynchronize_q();
	void hostSynchronize_p();
	void hostSynchronize_rho();

	void save(std::ofstream & fileSave)
	{		
		fileSave << get_N1() << "\n" << get_N2() << "\n" << get_N3() << "\n";
		fileSave << get_L1() << "\n" << get_L2() << "\n" << get_L3() << "\n";

		ifft();

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
	size_t get_N1() const { return N1; }
	size_t get_N2() const { return N2; }
	size_t get_N3() const { return N3; }
	size_t get_N3red() const { return N3red; }

	size_t get_N1_print() const { return N1_print; }
	size_t get_N2_print() const { return N2_print; }
	size_t get_N3_print() const { return N3_print; }

	double get_L1() const { return L1; }
	double get_L2() const { return L2; }
	double get_L3() const { return L3; }

	cudaRVector get_x1() const { return x1; }
	cudaRVector get_x2() const { return x2; }
	cudaRVector get_x3() const { return x3; }
	cudaRVector3 get_k1() const { return k1; }
	cudaRVector3 get_k2() const { return k2; }
	cudaRVector3 get_k3() const { return k3; }
	cudaRVector3 get_k_sqr() const { return k_sqr; }

	cudaRVector3 get_q() const { return q; }
	cudaRVector3 get_p() const { return p; }
	cudaRVector3 get_t() const { return t; }
	cudaCVector3 get_Q() const { return Q; }
	cudaCVector3 get_P() const { return P; }
	cudaCVector3 get_T() const { return T; }
	cudaRVector3 get_rho() const { return rho; }
	cudaCVector3 get_rhoK() const { return rhoK; }
	cudaRVector3 get_omega() const { return omega; }

	double get_time() const { return current_time; }
	double get_lambda() const { return lambda; }
	double get_g() const { return g; }

	/// Gets ptr
	double* get_x1_ptr() { return x1.get_Array(); }
	double* get_x2_ptr() { return x2.get_Array(); }
	double* get_x3_ptr() { return x3.get_Array(); }
	double* get_k1_ptr() { return k1.get_Array(); }
	double* get_k2_ptr() { return k2.get_Array(); }
	double* get_k3_ptr() { return k3.get_Array(); }

	double* get_k_sqr_ptr() { return k_sqr.get_Array(); }

	double* get_q_ptr() { return q.get_Array(); }
	double* get_p_ptr() { return p.get_Array(); }
	double* get_t_ptr() { return t.get_Array(); }
	complex* get_Q_ptr() { return Q.get_Array(); }
	complex* get_P_ptr() { return P.get_Array(); }
	complex* get_T_ptr() { return T.get_Array(); }

	complex* get_rhoK_ptr() const { return rhoK.get_Array(); }
	double* get_omega_ptr() const { return omega.get_Array(); }

	double* get_rho_ptr() { return rho.get_Array(); }

	double* get_buf_ptr() { return bufPrint.get_Array(); }


	/// FFT and IFFT
	void doFFT_t2T() { doFFTforward(t, T); }
	void doFFT_T2t() { doFFTinverce(T, t); };

	void doFFTforward(cudaCVector3 &f, cudaCVector3 &F) { cufft.forward(f, F); }
	void doFFTforward(cudaRVector3 &f, cudaCVector3 &F) { cufft.forward(f, F); }
	void doFFTinverce(cudaCVector3 &F, cudaCVector3 &f) { cufft.inverce(F, f); }
	void doFFTinverce(cudaCVector3 &F, cudaRVector3 &f) { cufft.inverce(F, f); }


	/// Sets 
	void set_lambda(const double _lambda) { lambda = _lambda; }
	void set_g(const double _g) { g = _g; }
	void setIFFTisNeeded() { isIFFTsync = false; }
	void setRhoCalcIsNeeded() { isRhoCalculateted = false; }
	void setSmthChanged() { 
		isRhoCalculateted = false; 
		isIFFTsync = false;
		isRhoKCalculateted = false;
	}


	/// Other methods 
	double get_dt(const double precision) const
	{
		return precision;
	}
	void set_sizes();
	void set_xk();
	void timestep(double dt) { current_time += dt; }
	void calculateRho();
	void calculateRhoK();
	void calculateOmega();

private:
	size_t N1, N2, N3, N3red;
	size_t N1_print, N2_print, N3_print;
	int factor;
	double L1, L2, L3;

	cudaRVector x1, x2, x3;
	cudaRVector3 k1, k2, k3;
	cudaRVector3 k_sqr;
	cudaRVector3 q, p, t;
	cudaCVector3 Q, P, T;

	cudaRVector3 rho;
	cudaCVector3 rhoK;
	cudaRVector3 omega;

	cudaRVector3 bufPrint;


	RVector3 RHost;
	CVector3 CHost;

	cuFFT cufft;

	double lambda, g;
	double current_time;
	bool isIFFTsync, isRhoCalculateted, isRhoKCalculateted;
};

