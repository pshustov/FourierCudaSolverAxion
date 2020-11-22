#include "interface.h"

#include <helper_cuda.h>
#include <string>
#include <iostream>
#include <typeinfo>


//------------ functions ------------//

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

int setCudaDev(int devIDin = -1)
{
	// devIDin = -1 to choose dev by hand

	int devID, numCudaDev;
	cudaDeviceProp deviceProps;

	cudaGetDeviceCount(&numCudaDev);

	if (devIDin == -1)
	{
		std::cout << "Available devices: " << std::endl;
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
				std::cout << "There is no available cuda device";
				throw;
			}
			else
			{
				devID = 0;
				checkCudaErrors(cudaSetDevice(devID));
			}
		}
	}
	else
	{
		if (devIDin < numCudaDev) {
			devID = devIDin;
			checkCudaErrors(cudaSetDevice(devID));
			checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
			std::cout << "Program runs on CUDA device " << devID << ": " << deviceProps.name << " with SM " << deviceProps.major << '.' << deviceProps.minor << std::endl;
		}
		else {
			std::cout << "Device number " << devIDin << " is not available. There are only " << numCudaDev << " available devices." << std::endl;
			throw;
		}

	}
	return devID;
}



//------------ Variable class ------------//

template<typename T>
Variable<T>::Variable(std::string _fullname, std::string _shortname) {
	is_var = false;
	fullname = _fullname;
	shortname = _shortname;
}

template<typename T>
Variable<T>::Variable(std::string _fullname, std::string _shortname, T defVal)
{
	var = defVal;
	is_var = true;
	fullname = _fullname;
	shortname = _shortname;
}

template<typename T>
void Variable<T>::set(T _var)
{
	var = _var;
	is_var = true;
}

template<typename T>
void Variable<T>::setOnStr(std::string strvar)
{
	std::cerr << "Variable type not recognized" << std::endl;
	throw;
}
template <> void Variable<bool>::setOnStr(std::string strvar)
{
	set(std::stoi(strvar));
}
template <> void Variable<char>::setOnStr(std::string strvar)
{
	set(std::stoi(strvar));
}
template <> void Variable<int>::setOnStr(std::string strvar)
{
	set(std::stoi(strvar));
}
template <> void Variable<float>::setOnStr(std::string strvar)
{
	set(std::stof(strvar));
}
template <> void Variable<double>::setOnStr(std::string strvar)
{
	set(std::stod(strvar));
}
template <> void Variable<std::string>::setOnStr(std::string strvar)
{
	set(strvar);
}
template <> void Variable<std::chrono::seconds>::setOnStr(std::string strvar)
{
	set(std::chrono::minutes{ std::stoi(strvar) });
}

template<typename T>
inline bool Variable<T>::isName(std::string name) {
	return (name == "--" + fullname) || (name == "-" + shortname);
}

template<typename T>
std::string Variable<T>::getName()
{
	return fullname;
}

template<typename T>
inline bool Variable<T>::isSet()
{
	return is_var;
}



//------------ Params class ------------//

Params::Params(int argc, char* argv[], bool isDebug) :
	var_isLoad(			"isLoad",			"iL",	false),
	var_isPrintVTK(		"isPrintVTK",		"iP",	false),
	var_timeLimitSave(	"timeLimitSave",	"tLS",	std::chrono::minutes{ 15 }),
	var_filename(		"filename",			"fn",	"in.txt"),
	var_gpuID(			"gpuID",			"gID",	0),
	var_precision(		"precision",		"p"),
	var_tau(			"tau",				"t"),
	var_lambda(			"lambda",			"l"),
	var_g(				"g",				"g")
{
	if (argc > 1)
	{
		int smthBeenSet = 1; // 1 - nothing been set, and 0 - smth been set
		for (int i = 1; i < argc;) {
			// 9 params

			smthBeenSet *= setVarOnArg(i, argc, argv, var_timeLimitSave);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_filename);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_isLoad);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_isPrintVTK);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_gpuID);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_precision);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_tau);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_lambda);
			smthBeenSet *= setVarOnArg(i, argc, argv, var_g);
			if (smthBeenSet == 1)
			{
				std::cerr << "Parameter " << argv[i] << "was not recognized" << std::endl;
				throw;
			}

		}
	}
	else
	{
		if (isDebug)
		{
			var_timeLimitSave.set(std::chrono::minutes{ 5 });
			var_filename.set("in.txt");
			var_isLoad.set(false);
			var_isPrintVTK.set(false);
			var_gpuID.set(0);
			var_precision.set(0.05);
			var_tau.set(25);
			var_lambda.set(0.0001);
			var_g.set(0);
		}
		else
		{
			myInterface();
		}
	}

	if (!isAllSet())
	{
		std::cout << "Not all parameters are set..." << std::endl;
		throw;
	}
	else
	{
		std::cout << "All parameters have been set..." << std::endl;
		if (var_isLoad.get()) {
			std::cout << "Loading ";
		}
		else {
			std::cout << "New task ";
		}
		std::cout << "from file \"" << var_filename.get() << "\"" << std::endl;
		setCudaDev(var_gpuID.get());
		std::cout << "lambda = " << var_lambda.get() << std::endl;
		std::cout << "g = " << var_g.get() << std::endl;
		std::cout << "dt = " << var_precision.get() << std::endl;
		std::cout << "tau = " << var_tau.get() << std::endl;

		std::cout << "Autosave every " << var_timeLimitSave.get().count() / 60 << " minutes" << std::endl;
		if (var_isPrintVTK.get()) {
			std::cout << "VTK files will be printed." << std::endl;
		}
		else {
			std::cout << "VTK files will not be printed." << std::endl;
		}

	}
}

template <typename T>
int Params::setVarOnArg(int& i, const int& argc, char* argv[], Variable<T>& var)
{
	if (i < argc) {
		std::string arg = argv[i];
		
		if (var.isName(arg))
		{
			if (i + 1 < argc) {
				var.setOnStr(argv[++i]);
				++i;
			}
			else {
				std::cerr << "--" << var.getName() << " option requires one argument." << std::endl;
				throw;
			}
			return 0;
		}
	}
	return 1;
}

void Params::myInterface()
{
	real temp;

	var_gpuID.set(setCudaDev());
	var_timeLimitSave.set(std::chrono::minutes{ 15 });

	var_isLoad.set(makeChoiceYN("New task?"));

	if (var_isLoad.get()) {
		var_filename.set("saveGrid.asv");
		var_lambda.set(0);
		var_g.set(0);
	}
	else {
		var_filename.set("in.txt");
		std::cout << "lambda = "; std::cin >> temp; var_lambda.set(temp);
		std::cout << "g = "; std::cin >> temp; var_g.set(temp);
	}

	std::cout << "precision = "; std::cin >> temp; var_precision.set(temp);
	std::cout << "tau = "; std::cin >> temp; var_tau.set(temp);	
	var_isPrintVTK.set(makeChoiceYN("VTK printing?"));
}

bool Params::isAllSet()
{
	// 9 params
	return var_precision.isSet() && var_tau.isSet() && var_lambda.isSet() &&
		var_g.isSet() && var_timeLimitSave.isSet() && var_filename.isSet() &&
		var_isLoad.isSet() && var_isPrintVTK.isSet() && var_gpuID.isSet();
}
