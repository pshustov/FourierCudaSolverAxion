#include <string>
#include <chrono>
#include "system.h"
#ifdef _WIN64
#include <conio.h>
#else
#include "kbhit.h"
#endif // _win64

#include <helper_cuda.h>


using namespace std::chrono;

void myInterface(double& precision, double& tau, double& g, double& lambda, std::string& filename, bool& isLoad, bool& isPrintVTK);
void save(time_point<steady_clock>& timer, seconds timeLimitSave, systemEquCuda_3D& S);
bool makeChoiceYN(std::string strQuestion);

int main(int argc, char* argv[])
{
	bool isDebug = true;

	double precision, tau, lambda, g;
	bool isLoad, isPrintVTK;
	seconds timeLimitSave;
	std::string filename;
	duration<double, std::milli> duration;

	if (isDebug)
	{
		precision = 0.001;
		tau = 0.1;
		lambda = 0.0001;
		g = 0;
		timeLimitSave = minutes{ 10 };
		filename = "in.txt";
		isLoad = false;
		isPrintVTK = false;
	}
	else
	{
		myInterface(precision, tau, g, lambda, filename, isLoad, isPrintVTK);
		timeLimitSave = minutes{ 5 };
	}

	auto startSave = steady_clock::now();

	systemEquCuda_3D S(filename, precision, tau, lambda, g, isLoad);
	save(startSave, seconds{ 0 }, S);

	S.printingMaxVal();
	S.printingVTK(isPrintVTK);

	std::cout << "--- Program GPU started --- \n";
	auto start = steady_clock::now();
	size_t i = 0;
	do
	{
		std::cout << "Step started...\n";
		S.evaluate();
		std::cout << "Step completed\n\n";

		save(startSave, timeLimitSave, S);
		S.printingMaxVal();
		S.printingVTK(isPrintVTK);

		printf("Energy = %.4e\tDeltaE = %.4e\n", S.getEnergy(), S.getDelta());

		duration = std::chrono::duration_cast<milliseconds>(steady_clock::now() - start);
		std::cout << "step: " << i << "\t\tcalculated time: " << S.getTime() << "\t\texecution time: " << duration.count() / 1000 << "sec" << std::endl << std::endl;
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

		++i;
	} while (true);
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

void myInterface(double& precision, double& tau, double& g, double& lambda, std::string& filename, bool& isLoad, bool& isPrintVTK)
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
	isPrintVTK = makeChoiceYN("VTK printing?");
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

