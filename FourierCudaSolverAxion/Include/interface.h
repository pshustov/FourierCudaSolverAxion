#pragma once

#include "cudaComplex.h"
#include <chrono>

bool makeChoiceYN(std::string strQuestion);
int setCudaDev(int devIDin);

template <typename T>
class Variable
{
public:
	Variable(std::string _fullname, std::string _shortname);
	Variable(std::string _fullname, std::string _shortname, T defVal);
	~Variable() {}

	void set(T _var);
	void setOnStr(std::string strvar);
	bool isName(std::string name);
	std::string getName();
	bool isSet();
	T get() { return var; };

private:
	T var;
	bool is_var;
	std::string fullname;
	std::string shortname;
};

class Params
{
public:
	Params(int argc, char* argv[], bool isDebug = false);
	~Params() {}

	template <typename T>
	int setVarOnArg(int& i, const int& argc, char* argv[], Variable<T>& var);
	void myInterface();
	bool isAllSet();

	// 9 params
	bool isLoad() { 
		return var_isLoad.get(); 
	}
	bool isPrintVTK() { 
		return var_isPrintVTK.get(); 
	}
	int gpuID() {
		return var_gpuID.get();
	}
	real precision() { 
		return var_precision.get(); 
	}
	real tau() { 
		return var_tau.get(); 
	}
	real lambda() { 
		return var_lambda.get(); 
	}
	real g() { 
		return var_g.get(); 
	}
	std::chrono::seconds timeLimitSave() { 
		return var_timeLimitSave.get(); 
	}
	std::string filename() { 
		return var_filename.get(); 
	}

private:

	// 9 params

	// have default vaule
	Variable<std::chrono::seconds> var_timeLimitSave;
	Variable<std::string> var_filename;
	Variable<bool> var_isLoad;
	Variable<bool> var_isPrintVTK;
	Variable<int> var_gpuID;

	// must be set
	Variable<real> var_precision;
	Variable<real> var_tau;
	Variable<real> var_lambda;
	Variable<real> var_g;
};


