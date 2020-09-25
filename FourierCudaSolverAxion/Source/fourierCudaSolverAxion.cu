#include <string>
#include <chrono>
#include "system.h"
#ifdef _WIN64
#include <conio.h>
#else
#include "kbhit.h"
#endif // _win64

#include <helper_cuda.h>
#include "device_launch_parameters.h"


using namespace std::chrono;

void myInterface(real& precision, real& tau, real& g, real& lambda, std::string& filename, bool& isLoad, bool& isPrintVTK);
void save(time_point<steady_clock>& timer, seconds timeLimitSave, systemEquCuda_3D& S);
bool makeChoiceYN(std::string strQuestion);

__global__ void kernelForwardNorm1(const size_t size, const size_t N, const real L, real* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] * L / N;
	}
}
__global__ void kernelForwardNorm1(const size_t size, const size_t N, const real L, complex* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] * L / N;
	}
}
__global__ void kernelInverseNorm1(const size_t size, const size_t N, const real L, real* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] / L;
	}
}
__global__ void kernelInverseNorm1(const size_t size, const size_t N, const real L, complex* V)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		V[i] = V[i] / L;
	}
}

void getSumSqr(int N, real& c, real* df, real* f)
{
	cudaMemcpy(f, df, N * sizeof(real), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	c = 0;
	for (size_t i = 0; i < N; i++)
	{
		c += f[i] * f[i];
	}
}

void getSumSqr(int N1, int N2, int N3, real& c, complex* df, complex* f)
{
	int N3red = N3 / 2 + 1;
	int Nred = N1 * N2 * N3red;
	cudaMemcpy(f, df, Nred * sizeof(complex), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int ind = 0;
	c = 0;
	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N2; j++)
		{
			ind = (i * N2 + j) * N3red;
			c += f[ind].absSqr();
			for (int k = 1; k < N3red; k++)
			{
				ind = (i * N2 + j) * N3red + k;
				c += 2 * f[ind].absSqr();
			}
		}
	}
}


int main(int argc, char* argv[])
{
	int N = 1 << 24;
	int N1 = 1 << 8;
	int N2 = 1 << 8;
	int N3 = 1 << 8;
	int N3red = N3 / 2 + 1;
	int Nred = N1 * N2 * N3red;
	int n[3] = { N1, N2, N3 };

	real L = 1, V = L * L * L;

	real* f = new real[N];
	real* F = new real[2 * Nred];

	for (size_t i = 0; i < N; i++)
	{
		f[i] = 1;
	}

	real* df;
	real* dF;
	cudaMalloc(&df, N * sizeof(real));
	cudaMalloc(&dF, 2. * Nred * sizeof(real));
	cudaMemcpy(df, f, N * sizeof(real), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cufftHandle planD2Z, planZ2D;
	cufftPlan3d(&planD2Z, n[0], n[1], n[2], CUFFT_R2C);
	cufftPlan3d(&planZ2D, n[0], n[1], n[2], CUFFT_C2R);

	real c1, c2;

	int FFT_BLOCK_SIZE = 256;
	dim3 blockF(FFT_BLOCK_SIZE);
	dim3 gridF((static_cast<unsigned int>(N) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);
	dim3 blockI(FFT_BLOCK_SIZE);
	dim3 gridI((static_cast<unsigned int>(Nred) + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE);

	for (size_t i = 0; i < 10; i++)
	{
		//cufftExecR2C(planD2Z, (cufftReal*)df, (cufftComplex*)dF);
		cufftXtExec(planD2Z, df, dF, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		kernelForwardNorm1 <<< gridF, blockF >>> (Nred, N, V, dF);
		cudaDeviceSynchronize();

		getSumSqr(N, c1, df, f);
		getSumSqr(N1, N2, N3, c2, (complex*)dF, (complex*)F);
		std::cout << c1 << "\t" << c2 * N / V / V << std::endl;

		//cufftExecC2R(planZ2D, (cufftComplex*)dF, (cufftReal*)df);
		cufftXtExec(planZ2D, dF, df, CUFFT_INVERSE);
		//cufftExecZ2D(planZ2D, (cufftDoubleComplex*)dF.getArray(), (cufftDoubleReal*)df.getArray());
		cudaDeviceSynchronize();
		kernelInverseNorm1 <<<gridI, blockI >>> (N, N, V, df);
		cudaDeviceSynchronize();

		getSumSqr(N, c1, df, f);
		getSumSqr(N1, N2, N3, c2, (complex*)dF, (complex*)F);
		std::cout << c1 << "\t" << c2 * N / V / V << std::endl << std::endl;
	}

	return 0;
}


//int main(int argc, char* argv[])
//{
//	bool isDebug = true;
//
//	real precision, tau, lambda, g;
//	bool isLoad, isPrintVTK;
//	seconds timeLimitSave;
//	std::string filename;
//	duration<double, std::milli> duration;
//
//	if (isDebug)
//	{
//		precision = 0.001;
//		tau = 0.1;
//		lambda = 0.0001;
//		g = 0;
//		timeLimitSave = minutes{ 10 };
//		filename = "in.txt";
//		isLoad = false;
//		isPrintVTK = false;
//	}
//	else
//	{
//		myInterface(precision, tau, g, lambda, filename, isLoad, isPrintVTK);
//		timeLimitSave = minutes{ 5 };
//	}
//
//	auto startSave = steady_clock::now();
//
//	systemEquCuda_3D S(filename, precision, tau, lambda, g, isLoad);
//	save(startSave, seconds{ 0 }, S);
//
//	S.printingMaxVal();
//	S.printingVTK(isPrintVTK);
//
//	std::cout << "--- Program GPU started --- \n";
//	auto start = steady_clock::now();
//	size_t i = 0;
//	do
//	{
//		std::cout << "Step started...\n";
//		S.evaluate();
//		std::cout << "Step completed\n\n";
//
//		save(startSave, timeLimitSave, S);
//		S.printingMaxVal();
//		S.printingVTK(isPrintVTK);
//
//		printf("Energy = %.4e\tDeltaE = %.4e\n", S.getEnergy(), S.getDelta());
//
//		duration = std::chrono::duration_cast<milliseconds>(steady_clock::now() - start);
//		std::cout << "step: " << i << "\t\tcalculated time: " << S.getTime() << "\t\texecution time: " << duration.count() / 1000 << "sec" << std::endl << std::endl;
//		std::cout << "------------------------------------------------------------------\n\n";
//
//
//
//		// keyboard signal processing
//		if (_kbhit()) {
//			switch (_getch()) {
//			case 's':
//				save(startSave, seconds{ 0 }, S);
//				break;
//			case 'q':
//				if (makeChoiceYN("Are you sure you want to save and finish the calculation?")) {
//					save(startSave, seconds{ 0 }, S);
//					return 0;
//				}
//				break;
//			case 'e':
//				if (makeChoiceYN("Are you sure you want to exit WITHOUT saving?")) {
//					return 0;
//				}
//				break;
//			default:
//				break;
//			}
//		}
//
//		++i;
//	} while (true);
//}

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

void myInterface(real& precision, real& tau, real& g, real& lambda, std::string& filename, bool& isLoad, bool& isPrintVTK)
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

