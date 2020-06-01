#include "testClass.h"

template <typename T>
void setData(int N, T* data)
{
	for (int i = 0; i < N; i++)
	{
		data[i] = i;
	}
}

template void setData<double>(int N, double* data);