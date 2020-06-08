#pragma once

#include "stdafx.h"

class cudaGrid_3D
{
public:
	cudaGrid_3D(const std::string filename);
	~cudaGrid_3D();

	void fft();
	void ifft();
	void ifftQ(bool isNormed = true);
	void ifftP(bool isNormed = true);

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

	/// Gets ptr
	double* get_x1_ptr() { return x1.getArray(); }
	double* get_x2_ptr() { return x2.getArray(); }
	double* get_x3_ptr() { return x3.getArray(); }
	double* get_k1_ptr() { return k1.getArray(); }
	double* get_k2_ptr() { return k2.getArray(); }
	double* get_k3_ptr() { return k3.getArray(); }

	double* get_k_sqr_ptr() { return k_sqr.getArray(); }

	double* get_q_ptr() { return q.getArray(); }
	double* get_p_ptr() { return p.getArray(); }
	double* get_t_ptr() { return t.getArray(); }
	complex* get_Q_ptr() { return Q.getArray(); }
	complex* get_P_ptr() { return P.getArray(); }
	complex* get_T_ptr() { return T.getArray(); }


	/// FFT and IFFT
	void doFFT_t2T() { doFFTforward(t, T); }
	void doFFT_T2t() { doFFTinverce(T, t); };

	void doFFTforward(cudaCVector3 &f, cudaCVector3 &F, bool isNormed = true) { cufft.forward(f, F, isNormed); }
	void doFFTforward(cudaRVector3 &f, cudaCVector3 &F, bool isNormed = true) { cufft.forward(f, F, isNormed); }
	void doFFTinverce(cudaCVector3 &F, cudaCVector3 &f, bool isNormed = true) { cufft.inverce(F, f, isNormed); }
	void doFFTinverce(cudaCVector3 &F, cudaRVector3 &f, bool isNormed = true) { cufft.inverce(F, f, isNormed); }


	/// Sets 
	void set_lambda(const double _lambda) { lambda = _lambda; }
	void set_g(const double _g) { g = _g; }
	void setSmthChanged() { 
		isEnergyCalculateted = false; 
		isIFFTsyncQ = false;
		isIFFTsyncP = false;
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

	void setcCUFFTstream(cudaStream_t stream) { cufft.setStreamAll(stream); }

private:
	size_t N1, N2, N3, N3red;
	double L1, L2, L3;

	cudaRVector x1, x2, x3;
	cudaRVector3 k1, k2, k3;
	cudaRVector3 k_sqr;
	cudaRVector3 q, p, t;
	cudaCVector3 Q, P, T;

	cuFFT cufft;

	double lambda, g;
	double current_time;
	bool isIFFTsyncQ, isIFFTsyncP, isEnergyCalculateted;

	double energy;

};

