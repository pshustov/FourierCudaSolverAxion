#pragma once
#include "cudaComplex.h"
#include "cudaReduction.h"

////////////		1D Vectors		////////////
template <typename T> class vector;
template <typename T> class cudaVector;
template <typename T> class cudaVectorDev;


/// <summary>
/// Template for 1D host vector
/// </summary>
/// <typeparam name="T">Vector of this type</typeparam>
template <typename T>
class vector
{
public:
	explicit vector(const size_t _N = 1) : N(_N)
	{
		Array = new T[N]();
	}
	vector(const vector& _V) : N(_V.getN())
	{
		Array = new T[N];
		for (size_t i = 0; i < N; i++)
			Array[i] = _V(i);
	}
	vector& operator=(const vector& _V)
	{
		if (this != &_V)
		{
			delete[] Array;
			N = _V.N;
			Array = new T[N];
			for (size_t i = 0; i < N; i++)
				Array[i] = _V(i);
		}
		return *this;
	}
	~vector()
	{
		delete[] Array;
	}

	vector(const cudaVector<T>& _V) : N(_V.getN())
	{
		Array = new T[N];
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToHost);
	}
	vector& operator=(cudaVector<T>& _V)
	{
		delete[] Array;
		N = _V.getN();
		Array = new T[N];
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToHost);
		return *this;
	}

	//TODO сделать вариант с проеркой условия i<N и посмотреть насколько это замедлит программу
	T& operator() (size_t i) { return Array[i]; }
	const T& operator() (size_t i) const { return Array[i]; }

	friend std::ostream& operator<< (std::ostream& os, vector<T>& v)
	{
		for (size_t i = 0; i < v.N - 1; i++)
		{
			os << v(i) << '\t';
		}
		os << v(v.N - 1);
		return os;
	}

	size_t getN() const { return N; }

	void set(const size_t _N) {
		delete[] Array;
		N = _N;
		Array = new T[N]();
	}

	vector<T>& operator+= (const T &b) {
		for (size_t i = 0; i < N; i++)
		{
			Array[i] += b;
		}
		return *this;
	}
	vector<T>& operator-= (const T &b) {
		for (size_t i = 0; i < N; i++)
		{
			Array[i] -= b;
		}
		return *this;
	}
	vector<T>& operator/= (const T &b) {
		for (size_t i = 0; i < N; i++)
		{
			Array[i] /= b;
		}
		return *this;
	}
	vector<T>& operator*= (const T &b) {
		for (size_t i = 0; i < N; i++)
		{
			Array[i] *= b;
		}
		return *this;
	}

	vector<T>& operator+= (const vector<T> &B)
	{
		if (N != B.getN())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] += B(i);
		}
		return *this;
	}
	vector<T>& operator-= (const vector<T> &B)
	{
		if (N != B.getN())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] -= B(i);
		}
		return *this;
	}
	vector<T>& operator*= (const vector<T> &B)
	{
		if (N != B.getN())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] *= B(i);
		}
		return *this;
	}
	vector<T>& operator/= (const vector<T> &B)
	{
		if (N != B.getN())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] /= B(i);
		}
		return *this;
	}

	friend vector<T> operator+(const vector<T> &A, const vector<T> &B)
	{
		if (A.getN() != B.getN())
			throw;

		vector<T> temp = A;
		temp += B;
		return temp;
	}
	friend vector<T> operator-(const vector<T> &A, const vector<T> &B)
	{
		if (A.getN() != B.getN())
			throw;

		vector<T> temp = A;
		temp -= B;
		return temp;
	}
	friend vector<T> operator*(const vector<T> &A, const vector<T> &B)
	{
		if (A.getN() != B.getN())
			throw;

		vector<T> temp = A;
		temp *= B;
		return temp;
	}
	friend vector<T> operator/(const vector<T> &A, const vector<T> &B)
	{
		if (A.getN() != B.getN())
			throw;

		vector<T> temp = A;
		temp /= B;
		return temp;
	}

	friend vector<T> operator+(const T &a, const vector<T> &B)
	{
		vector<T> temp(B.getN());
		for (size_t i = 0; i < B.getN(); i++)
		{
			temp(i) = a + B(i);
		}
		return temp;
	}
	friend vector<T> operator+(const vector<T> &A, const T &b)
	{
		vector<T> temp = A;
		temp += b;
		return temp;
	}
	friend vector<T> operator-(const T &a, const vector<T> &B)
	{
		vector<T> temp(B.getN());
		for (size_t i = 0; i < B.getN(); i++)
		{
			temp(i) = a - B(i);
		}
		return temp;
	}
	friend vector<T> operator-(const vector<T> &A, const T &b)
	{
		vector<T> temp = A;
		temp -= b;
		return temp;
	}
	friend vector<T> operator*(const T &a, const vector<T> &B)
	{
		vector<T> temp(B.getN());
		for (size_t i = 0; i < B.getN(); i++)
		{
			temp(i) = a * B(i);
		}
		return temp;
	}
	friend vector<T> operator*(const vector<T> &A, const T &b)
	{
		vector<T> temp = A;
		temp *= b;
		return temp;
	}
	friend vector<T> operator/(const vector<T> &A, const T &b)
	{
		vector<T> temp = A;
		temp /= b;
		return temp;
	}

	friend class cudaVector<T>;

private:
	size_t N;
	T* Array;
};
using RVector = vector<real>;
using CVector = vector<complex>;
//typedef vector<real> RVector;
//typedef vector<complex> CVector;


/// <summary>
/// Template for 1D cuda vector
/// </summary>
/// <typeparam name="T">Vector of this type</typeparam>
template <typename T>
class cudaVector
{
public:
	__host__ explicit cudaVector(const size_t _N = 1) : N(_N)
	{
		cudaMalloc(&Array, N * sizeof(T));
	}
	__host__ cudaVector(const cudaVector& _V) : N(_V.getN())
	{
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToDevice);

	}
	__host__ cudaVector& operator=(const cudaVector& _V)
	{
		if (this != &_V)
		{
			N = _V.N;

			cudaFree(Array);
			cudaMalloc(&Array, N * sizeof(T));
			cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToDevice);
		}
		return *this;
	}
	__host__ ~cudaVector()
	{
		cudaFree(Array);
	}

	__host__ cudaVector(const vector<T>& _V) : N(_V.getN())
	{
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ cudaVector& operator=(const vector<T>& _V)
	{
		N = _V.getN();

		cudaFree(Array);
		cudaMalloc(&Array, N * sizeof(T));
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyHostToDevice);

		return *this;
	}

	__host__ size_t getN() const { return N; }

	__host__ void set(const size_t _N)
	{
		N = _N;
		cudaFree(Array);
		cudaMalloc(&Array, N * sizeof(T));
	}

	__host__ T* getArray() { return Array; }

	__host__ T getSum(cudaStream_t stream) { return reductionSum<T>(N, Array, stream); }
	__host__ T getMax(cudaStream_t stream) { return reductionMax<T>(N, Array, stream); }

	friend class vector<T>;
	friend class cudaVectorDev<T>;


private:
	size_t N;
	T* Array;
};
using cudaRVector = cudaVector<real>;
using cudaCVector = cudaVector<complex>;
//typedef cudaVector<real> cudaRVector;
//typedef cudaVector<complex> cudaCVector;


/// <summary>
/// Template for 1D device vector
/// </summary>
/// <typeparam name="T">Vector of this type</typeparam>
template <typename T>
class cudaVectorDev
{
public:
	__host__ cudaVectorDev(const cudaVector<T>& _V) :N(_V.getN())
	{
		Array = _V.Array;
	}
	__host__ ~cudaVectorDev() { }

	__device__ size_t getN() const { return N; }
	__device__ T& operator() (size_t i) { return Array[i]; }
	__device__ const T& operator() (size_t i) const { return Array[i]; }

	friend class cudaVector<T>;

private:
	size_t N;
	T* Array;
};
using cudaRVectorDev = cudaVectorDev<real>;
using cudaCVectorDev = cudaVectorDev<complex>;
//typedef cudaVectorDev<real> cudaRVectorDev;
//typedef cudaVectorDev<complex> cudaCVectorDev;






////////////		3D Vectors		////////////
template <typename T> class vector3;
template <typename T> class cudaVector3;
template <typename T> class cudaVector3Dev;


/// <summary>
/// Template for 3D host vector
/// </summary>
/// <typeparam name="T">Vector of this type</typeparam>
template <typename T>
class vector3
{
public:
	explicit vector3(const size_t _N1 = 1, const size_t _N2 = 1, const size_t _N3 = 1) : N1(_N1), N2(_N2), N3(_N3)
	{
		Array = new T[N1*N2*N3]();
	}
	vector3(const vector3& _M) : N1(_M.getN1()), N2(_M.getN2()), N3(_M.getN3())
	{
		Array = new T[N1*N2*N3];
		for (size_t i = 0; i < N1*N2*N3; i++)
			Array[i] = _M(i);
	}
	vector3& operator=(const vector3& _M)
	{
		if (this != &_M)
		{
			delete[] Array;
			N1 = _M.getN1();
			N2 = _M.getN2();
			N3 = _M.getN3();
			Array = new T[N1*N2*N3];
			for (size_t i = 0; i < N1*N2*N3; i++)
				Array[i] = _M(i);
		}
		return *this;
	}
	~vector3()
	{
		delete[] Array;
	}

	vector3(const cudaVector3<T>& _V) : N1(_V.getN1()), N2(_V.getN2()), N3(_V.getN3())
	{
		Array = new T[N1*N2*N3]();
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyDeviceToHost);
	}
	vector3& operator=(const cudaVector3<T>& _V)
	{
		delete[] Array;
		N1 = _V.getN1();
		N2 = _V.getN2();
		N3 = _V.getN3();
		Array = new T[N1*N2*N3];
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		return *this;
	}

	T& operator() (size_t i) { return Array[i]; }
	const T& operator() (size_t i) const { return Array[i]; }

	T& operator() (size_t i, size_t j, size_t k) { return Array[(i*N2 + j)*N3 + k]; }
	const T& operator() (size_t i, size_t j, size_t k) const {
		if (i < N1 && j < N2 && k < N3)
		{
			return Array[(i*N2 + j)*N3 + k];
		}
		else
		{
			throw;
		}
	}

	void copyFromCudaPtr(const T *_Array) 
	{
		cudaMemcpy(Array, _Array, N1*N2*N3 * sizeof(T), cudaMemcpyDeviceToHost);
	}

	T sum() const {
		T summa = 0;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			summa += Array[i];
		}
		return summa;
	}

	friend std::ostream& operator<< (std::ostream& os, vector3<T>& _M)
	{
		for (size_t i = 0; i < _M.N1*_M.N2*_M.N3-1; i++)
		{
			os << _M(i) << '\t';
		}
		os << _M(_M.N1*_M.N2*_M.N3 - 1);
		return os;
	}

	size_t getN1() const { return N1; }
	size_t getN2() const { return N2; }
	size_t getN3() const { return N3; }
	
	size_t size() const { return N1*N2*N3; }

	//reshape and flash
	void set(const size_t _N1, const size_t _N2, const size_t _N3) {
		delete[] Array;
		N1 = _N1;
		N2 = _N2;
		N3 = _N3;
		Array = new T[N1*N2*N3]();
	}
	
	vector3<T>& operator+= (const T &b) {
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] += b;
		}
		return *this;
	}
	vector3<T>& operator-= (const T &b) {
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] -= b;
		}
		return *this;
	}
	vector3<T>& operator/= (const T &b) {
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] /= b;
		}
		return *this;
	}
	vector3<T>& operator*= (const T &b) {
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] *= b;
		}
		return *this;
	}

	vector3<T>& operator+= (const vector3<T> &B)
	{
		if (N1 != B.getN1() || N2 != B.getN2() || N3 != B.getN3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] += B(i);
		}
		return *this;
	}
	vector3<T>& operator-= (const vector3<T> &B)
	{
		if (N1 != B.getN1() || N2 != B.getN2() || N3 != B.getN3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] -= B(i);
		}
		return *this;
	}
	vector3<T>& operator*= (const vector3<T> &B)
	{
		if (N1 != B.getN1() || N2 != B.getN2() || N3 != B.getN3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] *= B(i);
		}
		return *this;
	}
	vector3<T>& operator/= (const vector3<T> &B)
	{
		if (N1 != B.getN1() || N2 != B.getN2() || N3 != B.getN3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] /= B(i);
		}
		return *this;
	}

	friend vector3<T> operator+(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.getN1() != B.getN1() || A.getN2() != B.getN2() || A.getN3() != B.getN3())
			throw;

		vector3<T> temp = A;
		temp += B;
		return temp;
	}
	friend vector3<T> operator-(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.getN1() != B.getN1() || A.getN2() != B.getN2() || A.getN3() != B.getN3())
			throw;

		vector3<T> temp = A;
		temp -= B;
		return temp;
	}
	friend vector3<T> operator*(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.getN1() != B.getN1() || A.getN2() != B.getN2() || A.getN3() != B.getN3())
			throw;

		vector3<T> temp = A;
		temp *= B;
		return temp;
	}
	friend vector3<T> operator/(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.getN1() != B.getN1() || A.getN2() != B.getN2() || A.getN3() != B.getN3())
			throw;

		vector3<T> temp = A;
		temp /= B;
		return temp;
	}

	friend vector3<T> operator+(const T &a, const vector3<T> &B)
	{
		vector3<T> temp(B.getN1(), B.getN2(), B.getN3());
		for (size_t i = 0; i < B.getN1()*B.getN2()*B.getN3(); i++)
		{
			temp(i) = a + B(i);
		}
		return temp;
	}
	friend vector3<T> operator+(const vector3<T> &A, const T &b)
	{
		vector3<T> temp = A;
		temp += b;
		return temp;
	}
	friend vector3<T> operator-(const T &a, const vector3<T> &B)
	{
		vector3<T> temp(B.getN1(), B.getN2(), B.getN3());
		for (size_t i = 0; i < B.getN1()*B.getN2()*B.getN3(); i++)
		{
			temp(i) = a - B(i);
		}
		return temp;
	}
	friend vector3<T> operator-(const vector3<T> &A, const T &b)
	{
		vector3<T> temp = A;
		temp -= b;
		return temp;
	}
	friend vector3<T> operator*(const T &a, const vector3<T> &B)
	{
		vector3<T> temp(B.getN1(), B.getN2(), B.getN3());
		for (size_t i = 0; i < B.getN1()*B.getN2()*B.getN3(); i++)
		{
			temp(i) = a * B(i);
		}
		return temp;
	}
	friend vector3<T> operator*(const vector3<T> &A, const T &b)
	{
		vector3<T> temp = A;
		temp *= b;
		return temp;
	}
	friend vector3<T> operator/(const vector3<T> &A, const T &b)
	{
		vector3<T> temp = A;
		temp /= b;
		return temp;
	}

	friend class cudaVector3<T>;

private:

	size_t N1, N2, N3;
	T* Array;
};
using RVector3 = vector3<real>;
using CVector3 = vector3<complex>;
//typedef vector3<real> RVector3;
//typedef vector3<complex> CVector3;


/// <summary>
/// Template for 3D cuda vector
/// </summary>
/// <typeparam name="T">Vector of this type</typeparam>
template <typename T>
class cudaVector3
{
public:

	__host__ explicit cudaVector3(const size_t _N1 = 1, const size_t _N2 = 1, const size_t _N3 = 1) : N1(_N1), N2(_N2), N3(_N3)
	{
		cudaMalloc(&Array, N1 * N2 * N3 * sizeof(T));
	}
	__host__ cudaVector3(const cudaVector3& _V) : N1(_V.getN1()), N2(_V.getN2()), N3(_V.getN3())
	{
		cudaMalloc(&Array, N1 * N2 * N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1 * N2 * N3 * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	__host__ cudaVector3& operator=(const cudaVector3& _V)
	{
		if (this != &_V)
		{
			N1 = _V.getN1();
			N2 = _V.getN2();
			N3 = _V.getN3();

			cudaFree(Array);
			cudaMalloc(&Array, N1 * N2 * N3 * sizeof(T));
			cudaMemcpy(Array, _V.Array, N1 * N2 * N3 * sizeof(T), cudaMemcpyDeviceToDevice);
		}
		return *this;
	}
	__host__ ~cudaVector3()
	{
		cudaFree(Array);
	}

	__host__ cudaVector3(const vector3<T>& _V) : N1(_V.getN1()), N2(_V.getN2()), N3(_V.getN3())
	{
		cudaMalloc(&Array, N1 * N2 * N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1 * N2 * N3 * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ cudaVector3& operator=(const vector3<T>& _V)
	{
		N1 = _V.getN1();
		N2 = _V.getN2();
		N3 = _V.getN3();

		cudaFree(Array);
		cudaMalloc(&Array, N1 * N2 * N3 * sizeof(T));
		cudaMemcpy(Array, _V.Array, N1 * N2 * N3 * sizeof(T), cudaMemcpyHostToDevice);

		return *this;
	}

	__host__ size_t getN1() const { return N1; }
	__host__ size_t getN2() const { return N2; }
	__host__ size_t getN3() const { return N3; }
	__host__ size_t size() const { return N1 * N2 * N3; }

	__host__ void set(const size_t _N1, const size_t _N2, const size_t _N3)
	{
		cudaFree(Array);
		N1 = _N1;
		N2 = _N2;
		N3 = _N3;
		cudaMalloc(&Array, N1 * N2 * N3 * sizeof(T));
	}

	__host__ T* getArray() { return Array; }

	__host__ T getSum(cudaStream_t stream) { return reductionSum<T>(static_cast<int>(size()), Array, stream); }
	__host__ T getMax(cudaStream_t stream) { return reductionMax<T>(static_cast<int>(size()), Array, stream); }

	friend class vector3<T>;
	friend class cudaVector3Dev<T>;

private:
	size_t N1, N2, N3;
	T* Array;
};
using cudaRVector3 = cudaVector3<real>;
using cudaCVector3 = cudaVector3<complex>;
//typedef cudaVector3<real> cudaRVector3;
//typedef cudaVector3<complex> cudaCVector3;


/// <summary>
/// Template for 3D device vector
/// </summary>
/// <typeparam name="T">Vector of this type</typeparam>
template <typename T>
class cudaVector3Dev
{
public:
	__host__ cudaVector3Dev(const cudaVector3<T>& _V) : N1(_V.getN1()), N2(_V.getN2()), N3(_V.getN3())
	{
		Array = _V.Array;
	}
	__host__ ~cudaVector3Dev() {}

	__device__ size_t size() const { return N1 * N2 * N3; }
	__device__ size_t getN1() const { return N1; }
	__device__ size_t getN2() const { return N2; }
	__device__ size_t getN3() const { return N3; }

	__device__ T& operator() (size_t i) { return Array[i]; }
	__device__ const T& operator() (size_t i) const { return Array[i]; }

	__device__ T& operator() (size_t i, size_t j, size_t k) { return Array[(i * N2 + j) * N3 + k]; }
	__device__ const T& operator() (size_t i, size_t j, size_t k) const { return Array[(i * N2 + j) * N3 + k]; }

	friend class cudaVector3<T>;

private:
	size_t N1, N2, N3;
	T* Array;
};
using cudaRVector3Dev = cudaVector3Dev<real>;
using cudaCVector3Dev = cudaVector3Dev<complex>;
//typedef cudaVector3Dev<real> cudaRVector3Dev;
//typedef cudaVector3Dev<complex> cudaCVector3Dev;