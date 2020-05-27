#include "stdafx.h"
#include <helper_cuda.h>

using namespace std::chrono;

void interface(double &precision, double &tau, double &g, size_t &N, double &lambda, std::string &filename, bool &isLoad);
void save(time_point<steady_clock> &timer, seconds timeLimitSave, systemEquCuda_3D &S);

int main(int argc, char *argv[])
{
	bool isDebug = true;

	size_t N;
	double precision, tau, lambda, g;
	bool isLoad;
	seconds timeLimitSave;
	std::string filename;
	duration<double, std::milli> duration;

	if (isDebug)
	{
		precision = 0.001;
		tau = 0.01;
		N = 100;
		lambda = -1;
		g = 0;
		timeLimitSave = minutes{ 10 };
		filename = "in.txt";
		isLoad = false;
	}
	else
	{
		interface(precision, tau, g, N, lambda, filename, isLoad);
		timeLimitSave = minutes{ 2 };
	}

	auto start = steady_clock::now();
	auto startSave = steady_clock::now();

	systemEquCuda_3D S(filename, precision, tau, lambda, g, isLoad);
	save(startSave, seconds{ 0 }, S);

	std::cout << "--- Program GPU started --- \n";

	for (size_t i = 0; i < N; i++)
	{
		std::cout << "Step started...\t";
		S.evaluate();
		std::cout << "Step completed\nPrinting started...\t";

		save(startSave, timeLimitSave, S);

		std::cout << "Printing done\n";
		printf("Energy = %.4e\tDeltaE = %.4e\n", S.get_energy(), S.get_delta());

		duration = std::chrono::duration_cast<milliseconds>(steady_clock::now() - start);
		std::cout << "step: " << i << "\t\tcalculated time: " << S.get_time()  << "\t\texecution time: " << duration.count()/1000 << "sec" << std::endl << std::endl;
		std::cout << "------------------------------------------------------------------\n\n";
	}
	
	std::cin >> tau;
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



//cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//cudaEventRecord(stop, 0); cudaEventSynchronize(stop);	float elapsedTime;	cudaEventElapsedTime(&elapsedTime, start, stop);
//std::cout << "cuFFT time: " << elapsedTime << "ms \n";	cudaEventDestroy(start); cudaEventDestroy(stop);

//std::clock_t startCPU;	double duration;	startCPU = std::clock();
//duration = (std::clock() - startCPU) / (double)CLOCKS_PER_SEC;	std::cout << "FFTW cpu time: " << duration*1000 << "ms \n";

