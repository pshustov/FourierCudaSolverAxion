#pragma once

#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

typedef double real;

class complex
{
public:
	__host__ __device__ complex(const real &_x = 0, const real &_y = 0) : x(_x), y(_y) { }
	__host__ __device__ ~complex() {}

	__host__ __device__ real get_real() const { return x; }
	__host__ __device__ real get_imag() const { return y; }

	__host__ __device__ real abs() const { return sqrt(x * x + y * y); }
	__host__ __device__ real absSqr() const { return x*x + y*y; }

	__host__ __device__ complex get_conj() const { return complex(x, -y); }

	__host__ __device__ void set(const real &_x, const real &_y) { x = _x; y = _y; }
	__host__ __device__ void set_polar(const real &_r, const real &_phi) { x = _r * cos(_phi); y = _r * sin(_phi); }

	__host__ __device__ friend complex operator-(const complex &a)
	{
		return complex(-a.x, -a.y);
	}

	__host__ __device__ friend complex operator+(const complex &a, const complex &b)
	{
		return complex(a.x + b.x, a.y + b.y);
	}
	__host__ __device__ friend complex operator-(const complex &a, const complex &b)
	{
		return complex(a.x - b.x, a.y - b.y);
	}
	__host__ __device__ friend complex operator*(const complex &a, const complex &b)
	{
		if (b.y == 0)
			return complex(a.x*b.x, a.y*b.x);
		if (a.y == 0)
			return complex(a.x*b.x, a.x*b.y);

		return complex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
	}
	__host__ __device__ friend complex operator/(const complex &a, const complex &b)
	{
		if (b.y == 0)
			return complex(a.x / b.x, a.y / b.x);

		return complex((a.x*b.x + a.y*b.y) / (b.x*b.x + b.y*b.y), (b.x*a.y - a.x*b.y) / (b.x*b.x + b.y*b.y));
	}

	__host__ __device__ complex& operator+=(const complex &b)
	{
		x += b.x;
		y += b.y;
		return *this;
	}
	__host__ __device__ complex& operator-=(const complex &b)
	{
		x -= b.x;
		y -= b.y;
		return *this;
	}
	__host__ __device__ complex& operator/=(const complex &b) {
		if (b.y == 0) {
			x /= b.x;
			y /= b.x;
			return *this;
		}
		else {
			real t_x = x;
			real abs = (b.x*b.x + b.y*b.y);
			x = (x*b.x + y*b.y) / abs;
			y = (b.x*y - t_x*b.y) / abs;
			return *this;
		}
	}
	__host__ __device__ complex& operator*=(const complex &b) {
		if (b.y == 0) {
			x *= b.x;
			y *= b.x;
			return *this;
		}
		else
		{
			real t_x = x;
			x = x*b.x - y*b.y;
			y = t_x*b.y + y*b.x;
			return *this;
		}
	}

	__host__ friend std::ostream& operator<<(std::ostream& os, complex c)
	{
		if (c.abs() == 0) os << "0";
		else if (c.get_imag() == 0) os << c.get_real();
		else if (c.get_real() == 0) os << c.get_imag() << "i";
		else os << c.get_real() << (c.get_imag() > 0 ? '+' : '-') << fabs(c.get_imag()) << "i";
		return os;
	}

private:
	real x;
	real y;
};
