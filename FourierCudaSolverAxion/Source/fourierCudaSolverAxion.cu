#include <string>
#include <chrono>
#include "system.h"
#include <helper_cuda.h>

#ifdef _WIN64
#include <conio.h>
#else
#include "kbhit.h"
#endif // _win64

using namespace std::chrono;

void save(time_point<steady_clock>& timer, seconds timeLimitSave, systemEquCuda_3D& S);

int main(int argc, char* argv[])
{
	bool isDebug = false;
	Params params(argc, argv, isDebug);

	duration<double, std::milli> duration;
	auto startSave = steady_clock::now();

	systemEquCuda_3D S(params);
	if (!params.isLoad()) {
		save(startSave, seconds{ 0 }, S);
	}

	S.printingMaxVal();
	S.printingVTK(params.isPrintVTK());

	std::cout << "--- Program GPU started --- \n";
	auto start = steady_clock::now();
	size_t i = 0;
	do
	{
		std::cout << "Step started...\n";
		S.evaluate();
		std::cout << "Step completed\n\n";

		save(startSave, params.timeLimitSave(), S);
		S.printingMaxVal();
		S.printingVTK(params.isPrintVTK());

		printf("Energy = %.4e\tDeltaE = %.4e\n", S.getEnergy(), S.getDelta0());

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
	} while (abs(S.getDelta1()) < 1);
	std::cout << "The program has been stopped due to energy conservation violation" << std::endl;
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


//cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//cudaEventRecord(stop, 0); cudaEventSynchronize(stop);	float elapsedTime;	cudaEventElapsedTime(&elapsedTime, start, stop);
//std::cout << "cuFFT time: " << elapsedTime << "ms \n";	cudaEventDestroy(start); cudaEventDestroy(stop);

//std::clock_t startCPU;	double duration;	startCPU = std::clock();
//duration = (std::clock() - startCPU) / (double)CLOCKS_PER_SEC;	std::cout << "FFTW cpu time: " << duration*1000 << "ms \n";

