#include "stdafx.h"
#include <helper_cuda.h>
#include <conio.h>

using namespace std::chrono;

void interface(double& precision, double& tau, double& g, size_t& N, double& lambda, std::string& filename, bool& isLoad);
void save(time_point<steady_clock>& timer, seconds timeLimitSave, systemEquCuda_3D& S);
bool makeChoiceYN(std::string strQuestion);

__global__ void kernelDev(cudaRVector3Dev A, cudaRVector3Dev B, cudaRVector3Dev C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < C.size())
	{
		C(i) = A(i) + B(i);
		//C(i) = 1;
	}
}

__global__ void kernelStd(int N, double* A, double* B, double* C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		C[i] = A[i] + B[i];
	}
}


int main(int argc, char* argv[])
{
	/*int N = 1 << 27;
	int blockSize = 256;
	RVector A(N), B(N), C(N);
	for (int i = 0; i < N; i++)
	{
		A(i) = i;
		B(i) = N - i;
		C(i) = 0;
	}
	cudaRVector cA(A), cB(B), cC(C);
	dim3 block(blockSize);
	dim3 grid((N + blockSize + 1) / blockSize);

	auto start = steady_clock::now();
	kernelStd<<<grid, block>>>(cA.getN(), cA.getArray(), cB.getArray(), cC.getArray());
	cudaDeviceSynchronize();
	microseconds duration = duration_cast<microseconds>(steady_clock::now() - start);
	std::cout << "Dev realis time: " << duration.count() * 1e-3 <<std::endl;

	start = steady_clock::now();
	kernelDev<<<grid, block>>>(cA, cB, cC);
	cudaDeviceSynchronize();
	duration = duration_cast<microseconds>(steady_clock::now() - start);
	std::cout << "Std realis time: " << duration.count() * 1e-3 <<std::endl;*/

	int blockSize = 256;

	std::string filename = "in.txt";
	cudaGrid_3D Grid(filename);

	dim3 block(blockSize);
	dim3 grid((Grid.size() + blockSize + 1) / blockSize);

	auto start = steady_clock::now();
	kernelDev<<<grid, block>>>(Grid.get_q(), Grid.get_p(), Grid.get_t());
	//kernelDev<<<grid, block>>>(Grid.q, Grid.p, Grid.t);
	//kernelStd<<<grid, block>>>(Grid.size(), Grid.get_q_ptr(), Grid.get_p_ptr(), Grid.get_t_ptr());
	cudaDeviceSynchronize();

	microseconds duration = duration_cast<microseconds>(steady_clock::now() - start);
	std::cout << "Dev grid time: " << duration.count() * 1e-3 << std::endl;

	/*bool isDebug = true;

	size_t N;
	double precision, tau, lambda, g;
	bool isLoad;
	seconds timeLimitSave;
	duration<double, std::milli> duration;

	if (isDebug)
	{
		precision = 0.001;
		tau = 0.1;
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

	auto startSave = steady_clock::now();

	systemEquCuda_3D S(filename, precision, tau, lambda, g, isLoad);
	save(startSave, seconds{ 0 }, S);

	std::cout << "--- Program GPU started --- \n";
	auto start = steady_clock::now();

	for (size_t i = 0; i < N; i++)
	{
		std::cout << "Step started...\t";
		S.evaluate();
		std::cout << "Step completed\nPrinting started...\t";

		save(startSave, timeLimitSave, S);

		std::cout << "Printing done\n";
		printf("Energy = %.4e\tDeltaE = %.4e\n", S.get_energy(), S.get_delta());

		duration = std::chrono::duration_cast<milliseconds>(steady_clock::now() - start);
		std::cout << "step: " << i << "\t\tcalculated time: " << S.get_time() << "\t\texecution time: " << duration.count() / 1000 << "sec" << std::endl << std::endl;
		std::cout << "------------------------------------------------------------------\n\n";

		// keyboard signal processing
		if (_kbhit()) {
			switch (_getch()) {
			case 's':
				save(startSave, seconds{ 0 }, S);
				break;
			case 'q':
				if (makeChoiceYN("Are you sure you want to save and finish the calculation?")) {
					save(startSave, seconds{ 0 }, S);
					return 0;
				}
				break;
			case 'e':
				if (makeChoiceYN("Are you sure you want to exit WITHOUT saving?")) {
					return 0;
				}
				break;
			default:
				break;
			}
		}

	}*/



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
		if (numCudaDev == 0)
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

void interface(double& precision, double& tau, double& g, size_t& N, double& lambda, std::string& filename, bool& isLoad)
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

void save(time_point<steady_clock>& timer, seconds timerLimit, systemEquCuda_3D& S)
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

