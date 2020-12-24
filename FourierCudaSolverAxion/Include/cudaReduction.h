#pragma once
template <typename T> T reductionSum(int size, T* inData, cudaStream_t stream);
template <typename T> T reductionMax(int size, T* inData, cudaStream_t stream);

template <typename T>
class Reduction
{
public:
	Reduction(int _size, T*& _Array, int _cpuFinalThreshold = 64, int _maxThreads = 256);
	~Reduction();

	void reset(int _size, int _cpuFinalThreshold = 64, int _maxThreads = 256);
	T getSum(cudaStream_t stream);
	T getMax(cudaStream_t stream);

private:
	int size;
	T*& Array;
	int cpuFinalThreshold, maxThreads;

	T* inData_dev;
	T* outData_dev;
	T* outData_host;

	void initialize();
	void clear();
};

