#include "stdafx.h"
#include <helper_cuda.h>
#include <conio.h>

using namespace std::chrono;

void interface(double &precision, double &tau, double &g, size_t &N, double &lambda, std::string &filename, bool &isLoad);
void save(time_point<steady_clock> &timer, seconds timeLimitSave, systemEquCuda_3D &S);
bool makeChoiceYN(std::string strQuestion);

__global__ void kernelTest_v1(int N, double* A)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		A[i] = -1;
	}
}

__global__ void kernelTest_v2(int N, d_cudaRVector A)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		A(i) = -1;
		if (i == 0)
			printf("\nAdress T: %p\n", (void*)&A);
		//printf("%g\n", A(i));
	}
}


class testClass
{
public:
	testClass(int _N): N(_N)
	{
		V.set_size_erase(N);
		for (int i = 0; i < N; i++)
		{
			V(i) = i;
			//std::cout << V(i) << std::endl;
		}

		cudaV = V;
		
		std::cout << "Adress of cudaV: " << &cudaV << std::endl;
	}
	~testClass() {};

	double getSum()
	{
		V = cudaV;
		double Sum = 0;
		for (int i = 0; i < N; i++)
		{
			Sum += V(i);
		}
		return Sum;
	}

	void testFun_v1()
	{
		int threads = 256;
		dim3 block(threads);
		dim3 grid((N + threads - 1) / threads);

		kernelTest_v1<<<grid, block>>>(N, cudaV.get_Array());
		cudaDeviceSynchronize();
	}

	void testFun_v2()
	{
		int threads = 256;
		dim3 block(threads);
		dim3 grid((N + threads - 1) / threads);

		kernelTest_v2<<<grid, block>>>(N, cudaV);
		cudaDeviceSynchronize();
	}

private:
	int N;
	RVector V;
	cudaRVector cudaV;

};


int main(int argc, char *argv[])
{
	int N = 1 << 20;
	testClass C(N);
	double Sum;

	duration<double, std::milli> dur1, dur2;

	//Sum = C.getSum();
	//std::cout << Sum << std::endl;

	//Sum = C.getSum();
	//std::cout << Sum << std::endl;

	int M = 1;

	auto start = steady_clock::now();
	for (int i = 0; i < M; i++)
	{
		C.testFun_v1();
	}
	dur1 = duration_cast<milliseconds>(steady_clock::now() - start);

	start = steady_clock::now();
	for (int i = 0; i < M; i++)
	{
		C.testFun_v2();
	}
	dur2 = duration_cast<milliseconds>(steady_clock::now() - start);

	std::cout << "dur1 = " << dur1.count() << "\tdur2 =" << dur2.count();

	return 0;
}

void setCudaDev()
{
	int devID, numCudaDev;
	cudaDeviceProp deviceProps;

	cudaGetDeviceCount(&numCudaDev);

	for (int i = 0; i < numCudaDev; i++)
	{
		checkCudaErrors(cudaGetDeviceProperties(&deviceProps, i));
		std::cout << "CUDA device " << i << ": " << deviceProps.name << " with SM " << deviceProps.major << '.' << deviceProps.minor << std::endl;
	}
	if (numCudaDev > 1)
	{
		std::cout << "Select cuda device: ";
		std::cin >> devID;
	}
	else
	{
		if (numCudaDev==0)
		{
			std::cout << "There is no cuda device";
			throw;
		}
		else
		{
			devID = 0;
			checkCudaErrors(cudaSetDevice(devID));
		}
	}

	checkCudaErrors(cudaSetDevice(devID));
}

void interface(double &precision, double &tau, double &g, size_t &N, double &lambda, std::string &filename, bool &isLoad)
{
	setCudaDev();

	int a;
	std::cout << "1. New task\n";
	std::cout << "2. Load task\n";

	bool isDone = false;
	do
	{
		std::cin >> a;
		switch (a)
		{
		case 1:
			isLoad = false;
			filename = "in.txt";
			std::cout << "lambda = ";
			std::cin >> lambda;
			std::cout << "g = ";
			std::cin >> g;
			isDone = true;
			break;
		case 2:
			isLoad = true;
			filename = "saveGrid.asv";
			isDone = true;
			break;
		default:

			break;
		}
	} while (!isDone);
	
	std::cout << "precision = ";
	std::cin >> precision;
	std::cout << "tau = ";
	std::cin >> tau;
	std::cout << "N = ";
	std::cin >> N;

}

void save(time_point<steady_clock> &timer, seconds timerLimit, systemEquCuda_3D &S)
{
	auto elapsedTime = std::chrono::duration_cast<seconds>(steady_clock::now() - timer);
	if (elapsedTime >= timerLimit)
	{
		std::cout << "\n\nSAVING...\t";
		S.save();
		timer = steady_clock::now();
		std::cout << "SAVED\n\n";
	}
}

bool makeChoiceYN(std::string strQuestion)
{
	char type;
	do {
		std::cout << strQuestion << " [y/n]: ";
		std::cin >> type;
	} while (!std::cin.fail() && type != 'y' && type != 'n');

	if (type == 'y')
		return true;
	else
		return false;
}

//cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//cudaEventRecord(stop, 0); cudaEventSynchronize(stop);	float elapsedTime;	cudaEventElapsedTime(&elapsedTime, start, stop);
//std::cout << "cuFFT time: " << elapsedTime << "ms \n";	cudaEventDestroy(start); cudaEventDestroy(stop);

//std::clock_t startCPU;	double duration;	startCPU = std::clock();
//duration = (std::clock() - startCPU) / (double)CLOCKS_PER_SEC;	std::cout << "FFTW cpu time: " << duration*1000 << "ms \n";

