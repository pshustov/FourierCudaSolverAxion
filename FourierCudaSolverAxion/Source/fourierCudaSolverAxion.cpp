#include "stdafx.h"
#include <helper_cuda.h>
#include <conio.h>

using namespace std::chrono;

void interface(double& precision, double& tau, double& g, size_t& N, double& lambda, std::string& filename, bool& isLoad);
void save(time_point<steady_clock>& timer, seconds timeLimitSave, systemEquCuda_3D& S);
bool makeChoiceYN(std::string strQuestion);


//__global__ void kernelSet(cudaRVectorDev A) 
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i < A.getN())
//	{
//		A(i) = cos(2 * Ma_PI * 3 * (double)i / A.getN()) + 2;
//	}
//
//}
//
////#define N 500000 
////#define NSTEP 1000
////#define NKERNEL 20
//
////__global__ void shortKernel(float* out_d, float* in_d) 
////{
////	int idx = blockIdx.x * blockDim.x + threadIdx.x;
////	if (idx < N) out_d[idx] = 1.23 * in_d[idx];
////}
//
////__global__ void kernelTest20(int N, double* a)
////{
////	int i = blockIdx.x * blockDim.x + threadIdx.x;
////	if (i < N)
////	{
////		a[i] = 1.01 * a[i];
////	}
////}
////
////__global__ void kernelTest21(int N, double* a, double* b)
////{
////	int i = blockIdx.x * blockDim.x + threadIdx.x;
////	if (i < N)
////	{
////		a[i] += b[i];
////	}
////}

int main(int argc, char* argv[])
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
		tau = 0.1;
		N = 100;
		lambda = 0;
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

	}

	/*int N = 1 << 12;

	double* a = new double[N];
	double* b = new double[N];

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 1;
	}

	double *da, *db;
	cudaMalloc(&da, N * sizeof(double));
	cudaMalloc(&db, N * sizeof(double));

	cudaMemcpy(da, a, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, N * sizeof(double), cudaMemcpyHostToDevice);

	std::ofstream fout("dataTest.txt");

	dim3 block(BLOCK_SIZE);
	dim3 grid((N + BLOCK_SIZE + 1) / BLOCK_SIZE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;


	cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
	for (int i = 0; i < 100; i++)
	{
		cudaMemcpyAsync(da, a, N * sizeof(double), cudaMemcpyHostToDevice, stream);
		kernelTest20<<< grid, block, 0, stream >>>(N, da);
		kernelTest21<<< grid, block, 0, stream >>>(N, da, db);
		cudaMemcpyAsync(a, da, N * sizeof(double), cudaMemcpyDeviceToHost, stream);
		std::cout << a[1] << std::endl;
	}
	//cudaStreamSynchronize(stream);
	cudaStreamEndCapture(stream, &graph);
	cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);


	cudaGraphLaunch(instance, stream);
	cudaStreamSynchronize(stream);
	cudaMemcpyAsync(a, da, N * sizeof(double), cudaMemcpyDeviceToHost, stream);

	for (int i = 0; i < N; i++)
	{
		fout << a[i] << '\n';
	}

	fout.close();*/
	
	/*double L = 3;
	int N = 1 << 12;

	int blockN = 256;
	std::ofstream fout("dataTest.txt");
	dim3 block(blockN);
	dim3 grid((N + blockN + 1) / blockN);
	RVector A(N), B(N);
	cudaRVector cA(A), cB(N + 2);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;

	cufftHandle plan;
	if (cufftPlan1d(&plan, N, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 1;
	}
	cufftSetStream(plan, stream);


	cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
	
	kernelSet<<<grid, block, 0, stream>>>(cA);
	cufftExecD2Z(plan, (cufftDoubleReal*)cA.getArray(), (cufftDoubleComplex*)cB.getArray());
	B = cB;
	//std::cout << B(0) << '\t' << B(6);
	
	cudaStreamEndCapture(stream, &graph);
	cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);


	cudaGraphLaunch(instance, stream);
	cudaStreamSynchronize(stream);

	B = cB;
	A = cA;
	for (int i = 0; i < N ; i++)
	{
		fout << B(i) << '\n';
	}
	fout.close();*/

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

