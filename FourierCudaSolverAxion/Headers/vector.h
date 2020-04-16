	#pragma once

#include "stdafx.h"

template <typename T> class cudaVector;
template <typename T> class cudaVector3;

template <typename T>
class vector
{
public:
	explicit vector(const size_t _N = 1) : N(_N)
	{
		Array = new T[N]();
	}
	vector(const vector& _V) : N(_V.get_N())
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

	vector(const cudaVector<T>& _V) : N(_V.get_N())
	{
		Array = new T[N];
		cudaMemcpy(Array, _V.Array, N * sizeof(T), cudaMemcpyDeviceToHost);
	}
	vector& operator=(cudaVector<T>& _V)
	{
		delete[] Array;
		N = _V.get_N();
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

	size_t get_N() const { return N; }

	void set_size_erase(const size_t _N) {
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
		if (N != B.get_N())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] += B(i);
		}
		return *this;
	}
	vector<T>& operator-= (const vector<T> &B)
	{
		if (N != B.get_N())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] -= B(i);
		}
		return *this;
	}
	vector<T>& operator*= (const vector<T> &B)
	{
		if (N != B.get_N())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] *= B(i);
		}
		return *this;
	}
	vector<T>& operator/= (const vector<T> &B)
	{
		if (N != B.get_N())
			throw;
		for (size_t i = 0; i < N; i++)
		{
			Array[i] /= B(i);
		}
		return *this;
	}

	friend vector<T> operator+(const vector<T> &A, const vector<T> &B)
	{
		if (A.get_N() != B.get_N())
			throw;

		vector<T> temp = A;
		temp += B;
		return temp;
	}
	friend vector<T> operator-(const vector<T> &A, const vector<T> &B)
	{
		if (A.get_N() != B.get_N())
			throw;

		vector<T> temp = A;
		temp -= B;
		return temp;
	}
	friend vector<T> operator*(const vector<T> &A, const vector<T> &B)
	{
		if (A.get_N() != B.get_N())
			throw;

		vector<T> temp = A;
		temp *= B;
		return temp;
	}
	friend vector<T> operator/(const vector<T> &A, const vector<T> &B)
	{
		if (A.get_N() != B.get_N())
			throw;

		vector<T> temp = A;
		temp /= B;
		return temp;
	}

	friend vector<T> operator+(const T &a, const vector<T> &B)
	{
		vector<T> temp(B.get_N());
		for (size_t i = 0; i < B.get_N(); i++)
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
		vector<T> temp(B.get_N());
		for (size_t i = 0; i < B.get_N(); i++)
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
		vector<T> temp(B.get_N());
		for (size_t i = 0; i < B.get_N(); i++)
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
using RVector = vector<double>;
using CVector = vector<complex>;


//template <typename T>
//class vector2
//{
//public:
//	//constructor
//	explicit vector2(const size_t _N1 = 1, const size_t _N2 = 1) : N1(_N1), N2(_N2)
//	{
//		//проверить происходит ли инициализация по умолчани у complex при таком задании
//		Array = new T[N1*N2]();
//	}
//	//copy constructor	вызывается, когда создаётся новый vector2 и для его инициализации берётся значение существующего vector2
//	vector2(const vector2& _M) : N1(_M.get_N1()), N2(_M.get_N2())
//	{
//		Array = new T[N1*N2];
//		for (size_t i = 0; i < N1*N2; i++)
//			Array[i] = _M(i);
//	}
//	//copy-assigment operator 
//	vector2& operator=(const vector2& _M)
//	{
//		if (this != &_M)
//		{
//			delete[] Array;
//			N1 = _M.get_N1();
//			N2 = _M.get_N2();
//			Array = new T[N1*N2];
//			for (size_t i = 0; i < N1*N2; i++)
//				Array[i] = _M(i);
//		}
//		return *this;
//	}
//	//destructor
//	~vector2()
//	{
//		delete[] Array;
//	}
//
//	T& operator() (size_t i) { return Array[i]; }
//	const T& operator() (size_t i) const { return Array[i]; }
//
//	T& operator() (size_t i, size_t j) { return Array[i*N2 + j]; }
//	const T& operator() (size_t i, size_t j) const {
//		//return ((i >= 0 && i < N1 && j >= 0 && j < N2) ? Array[i*N2 + j] : throw); 
//		if (i < N1 && j < N2)
//		{
//			return Array[i*N2 + j];
//		}
//		else
//		{
//			throw;
//		}
//	}
//
//	friend std::ostream& operator<< (std::ostream& os, vector2<T>& _M)
//	{
//		os << '{';
//		int i = -1;
//		while (++i < _M.N1)
//		{
//			os << " {";
//			int j = -1;
//			while (++j < _M.N2)
//			{
//				os << ' ';
//				os << _M(i, j);
//				if (j + 1 < _M.N2) os << ',';
//			}
//			os << " }\n";
//			if (i + 1 < _M.N1) os << ',';
//		}
//		os << " }";
//		return os;
//	}
//
//	const size_t get_N1() const {
//		return N1;
//	}
//	const size_t get_N2() const {
//		return N2;
//	}
//
//	//reshape and flash
//	void set_size_erase(const size_t _N1, const size_t _N2) {
//		delete[] Array;
//		N1 = _N1;
//		N2 = _N2;
//		Array = new T[N1*N2]();
//	}
//
//	void set(const vector<T> V, const int type, const size_t n)
//	{
//		switch (type)
//		{
//		case ROW:
//			if (n < N1)
//			{
//				for (size_t i = 0; i < N2; i++)
//				{
//					operator() (n, i) = V(i);
//				}
//			}
//			else
//			{
//				throw;
//			}
//			break;
//		case COLUMN:
//			if (n < N2)
//			{
//				for (size_t i = 0; i < N1; i++)
//				{
//					operator() (i, n) = V(i);
//				}
//			}
//			else
//			{
//				throw;
//			}
//			break;
//		default:
//			break;
//		}
//	}
//
//	vector2<T>& operator+= (const T &b) {
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] += b;
//		}
//		return *this;
//	}
//	vector2<T>& operator-= (const T &b) {
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] -= b;
//		}
//		return *this;
//	}
//	vector2<T>& operator/= (const T &b) {
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] /= b;
//		}
//		return *this;
//	}
//	vector2<T>& operator*= (const T &b) {
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] *= b;
//		}
//		return *this;
//	}
//
//	vector2<T>& operator+= (const vector2<T> &B)
//	{
//		if (N1 != B.get_N1() || N2 != B.get_N2())
//			throw;
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] += B(i);
//		}
//		return *this;
//	}
//	vector2<T>& operator-= (const vector2<T> &B)
//	{
//		if (N1 != B.get_N1() || N2 != B.get_N2())
//			throw;
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] -= B(i);
//		}
//		return *this;
//	}
//	vector2<T>& operator*= (const vector2<T> &B)
//	{
//		if (N1 != B.get_N1() || N2 != B.get_N2())
//			throw;
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] *= B(i);
//		}
//		return *this;
//	}
//	vector2<T>& operator/= (const vector2<T> &B)
//	{
//		if (N1 != B.get_N1() || N2 != B.get_N2())
//			throw;
//		for (size_t i = 0; i < N1*N2; i++)
//		{
//			Array[i] /= B(i);
//		}
//		return *this;
//	}
//
//	friend vector2<T> operator+(const vector2<T> &A, const vector2<T> &B)
//	{
//		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2())
//			throw;
//
//		vector2<T> temp = A;
//		temp += B;
//		return temp;
//	}
//	friend vector2<T> operator-(const vector2<T> &A, const vector2<T> &B)
//	{
//		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2())
//			throw;
//
//		vector2<T> temp = A;
//		temp -= B;
//		return temp;
//	}
//	friend vector2<T> operator*(const vector2<T> &A, const vector2<T> &B)
//	{
//		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2())
//			throw;
//
//		vector2<T> temp = A;
//		temp *= B;
//		return temp;
//	}
//	friend vector2<T> operator/(const vector2<T> &A, const vector2<T> &B)
//	{
//		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2())
//			throw;
//
//		vector2<T> temp = A;
//		temp /= B;
//		return temp;
//	}
//
//	friend vector2<T> operator+(const T &a, const vector2<T> &B)
//	{
//		vector2<T> temp(B.get_N1()*B.get_N2());
//		for (size_t i = 0; i < B.get_N1()*B.get_N2(); i++)
//		{
//			temp(i) = a + B(i);
//		}
//		return temp;
//	}
//	friend vector2<T> operator+(const vector2<T> &A, const T &b)
//	{
//		vector2<T> temp = A;
//		temp += b;
//		return temp;
//	}
//	friend vector2<T> operator-(const T &a, const vector2<T> &B)
//	{
//		vector2<T> temp(B.get_N1()*B.get_N2());
//		for (size_t i = 0; i < B.get_N1()*B.get_N2(); i++)
//		{
//			temp(i) = a - B(i);
//		}
//		return temp;
//	}
//	friend vector2<T> operator-(const vector2<T> &A, const T &b)
//	{
//		vector2<T> temp = A;
//		temp -= b;
//		return temp;
//	}
//	friend vector2<T> operator*(const T &a, const vector2<T> &B)
//	{
//		vector2<T> temp(B.get_N1()*B.get_N2());
//		for (size_t i = 0; i < B.get_N1()*B.get_N2(); i++)
//		{
//			temp(i) = a * B(i);
//		}
//		return temp;
//	}
//	friend vector2<T> operator*(const vector2<T> &A, const T &b)
//	{
//		vector2<T> temp = A;
//		temp *= b;
//		return temp;
//	}
//	friend vector2<T> operator/(const vector2<T> &A, const T &b)
//	{
//		vector2<T> temp = A;
//		temp /= b;
//		return temp;
//	}
//
//private:
//	size_t N1, N2;
//	T* Array;
//};
//using RVector2 = vector2<double>;
//using CVector2 = vector2<complex>;
//using CVector2 = vector2<complex>;


template <typename T>
class vector3
{
public:
	explicit vector3(const size_t _N1 = 1, const size_t _N2 = 1, const size_t _N3 = 1) : N1(_N1), N2(_N2), N3(_N3)
	{
		Array = new T[N1*N2*N3]();
	}
	vector3(const vector3& _M) : N1(_M.get_N1()), N2(_M.get_N2()), N3(_M.get_N3())
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
			N1 = _M.get_N1();
			N2 = _M.get_N2();
			N3 = _M.get_N3();
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

	vector3(const cudaVector3<T>& _V) : N1(_V.get_N1()), N2(_V.get_N2()), N3(_V.get_N3())
	{
		Array = new T[N1*N2*N3]();
		cudaMemcpy(Array, _V.Array, N1*N2*N3 * sizeof(T), cudaMemcpyDeviceToHost);
	}
	vector3& operator=(const cudaVector3<T>& _V)
	{
		delete[] Array;
		N1 = _V.get_N1();
		N2 = _V.get_N2();
		N3 = _V.get_N3();
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

	size_t get_N1() const { return N1; }
	size_t get_N2() const { return N2; }
	size_t get_N3() const { return N3; }
	
	size_t size() const { return N1*N2*N3; }

	//reshape and flash
	void set_size_erase(const size_t _N1, const size_t _N2, const size_t _N3) {
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
		if (N1 != B.get_N1() || N2 != B.get_N2() || N3 != B.get_N3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] += B(i);
		}
		return *this;
	}
	vector3<T>& operator-= (const vector3<T> &B)
	{
		if (N1 != B.get_N1() || N2 != B.get_N2() || N3 != B.get_N3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] -= B(i);
		}
		return *this;
	}
	vector3<T>& operator*= (const vector3<T> &B)
	{
		if (N1 != B.get_N1() || N2 != B.get_N2() || N3 != B.get_N3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] *= B(i);
		}
		return *this;
	}
	vector3<T>& operator/= (const vector3<T> &B)
	{
		if (N1 != B.get_N1() || N2 != B.get_N2() || N3 != B.get_N3())
			throw;
		for (size_t i = 0; i < N1*N2*N3; i++)
		{
			Array[i] /= B(i);
		}
		return *this;
	}

	friend vector3<T> operator+(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2() || A.get_N3() != B.get_N3())
			throw;

		vector3<T> temp = A;
		temp += B;
		return temp;
	}
	friend vector3<T> operator-(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2() || A.get_N3() != B.get_N3())
			throw;

		vector3<T> temp = A;
		temp -= B;
		return temp;
	}
	friend vector3<T> operator*(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2() || A.get_N3() != B.get_N3())
			throw;

		vector3<T> temp = A;
		temp *= B;
		return temp;
	}
	friend vector3<T> operator/(const vector3<T> &A, const vector3<T> &B)
	{
		if (A.get_N1() != B.get_N1() || A.get_N2() != B.get_N2() || A.get_N3() != B.get_N3())
			throw;

		vector3<T> temp = A;
		temp /= B;
		return temp;
	}

	friend vector3<T> operator+(const T &a, const vector3<T> &B)
	{
		vector3<T> temp(B.get_N1(), B.get_N2(), B.get_N3());
		for (size_t i = 0; i < B.get_N1()*B.get_N2()*B.get_N3(); i++)
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
		vector3<T> temp(B.get_N1(), B.get_N2(), B.get_N3());
		for (size_t i = 0; i < B.get_N1()*B.get_N2()*B.get_N3(); i++)
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
		vector3<T> temp(B.get_N1(), B.get_N2(), B.get_N3());
		for (size_t i = 0; i < B.get_N1()*B.get_N2()*B.get_N3(); i++)
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
using RVector3 = vector3<double>;
using CVector3 = vector3<complex>;