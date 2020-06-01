#include "testClass.h"

template <typename T>
void setData(int N, T* data);


double doSmth(int N, double* data)
{
	setData<double>(N, data);
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += data[i];
	}
	return sum;
}



template <typename T>
T doSmthTemplate(int N, T* data)
{
	setData<T>(N, data);
	T sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += data[i];
	}
	return sum;
}

template double doSmthTemplate<double>(int N, double* data);