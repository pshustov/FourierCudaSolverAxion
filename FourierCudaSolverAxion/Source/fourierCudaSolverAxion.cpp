#include "stdafx.h"
#include <helper_cuda.h>
#include <conio.h>

using namespace std::chrono;

void interface(double &precision, double &tau, double &g, size_t &N, double &lambda, std::string &filename, bool &isLoad);
void save(time_point<steady_clock> &timer, seconds timeLimitSave, systemEquCuda_3D &S);
bool makeChoiceYN(std::string strQuestion);

int main(int argc, char *argv[])
{
	int N = 1 << 16;
	RVector V(N);

	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		V(i) = i;
		sum += i;
	}

	cudaRVector cV(V);
	auto start = steady_clock::now();
	double sumCuda = cV.getSum();
	milliseconds duration = std::chrono::duration_cast<milliseconds>(steady_clock::now() - start);

	std::cout << "time " << duration.count() << "ms\n" << sum << '\n' << sumCuda;

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



//
//#include <cstdio>
//
//template <typename T>
//__global__ void kernel(T f) {
//	int i = 10;
//	printf("%d \t %d\n", i, f(i));
//}
//
//
//int main() {
//	int k = 13;
//
//	auto func_gpu = [=] __device__(int i) {
//		return i * i;
//	};
//
//	kernel<int> << <1, 32 >> > (func_gpu);
//	cudaDeviceSynchronize();
//
//}