#include "stdafx.h"

double reductionMax(int size, double *inData);
double reductionSum(int size, double *inData);

void systemEquCuda_3D::evaluate()
{
	double t = tau, dt;

	Grid.calculateRhoK();
	Grid.calculateOmega();
	distr.setDistributionFunctionAsync(Grid.get_time(), Grid.get_rhoK_ptr(), Grid.get_omega_ptr());

	int countIn = 0, countOut = 0;

	while (t >= (dt = Grid.get_dt(precision)) ) {
		evlulate_step(dt);
		++countOut;
		if (distr.isDistributionFunctionReady())
		{
			++countIn;

			Grid.calculateRhoK();
			Grid.calculateOmega();
			distr.setDistributionFunctionAsync(Grid.get_time(), Grid.get_rhoK_ptr(), Grid.get_omega_ptr());
		}

		t -= dt;
	}

	if (t < dt && t > 0) {
		evlulate_step(t);
	}

	distr.waitUntilAsyncEnd();

	isEnergyCalculated = false;

	std::cout << "Proportion of In/Out = " << float(countIn) / float(countOut) << std::endl;
}

void systemEquCuda_3D::printingVTK()
{
	double time = get_time();

	Grid.ifft();
	Grid.hostSynchronize_q();

	char buf[100];
	sprintf(buf, "dataFolder/data_%07.0f.vtk", time * 1000);
	std::ofstream outVTK(buf);
	outVTK.precision(4);

	size_t N1 = Grid.get_N1_print(), N2 = Grid.get_N2_print(), N3 = Grid.get_N3_print();
	double L1 = Grid.get_L1(), L2 = Grid.get_L2(), L3 = Grid.get_L3();

	outVTK << "# vtk DataFile Version 3.0\n";
	outVTK << "Density interpolation\n";
	outVTK << "ASCII\n";
	outVTK << "DATASET STRUCTURED_POINTS\n";
	outVTK << "DIMENSIONS " << N1 << " " << N2 << " " << N3 << "\n";
	outVTK << "ORIGIN 0 0 0\n";
	outVTK << "SPACING " << L1 / N1 << " " << L2 / N2 << " " << L3 / N3 << "\n";
	outVTK << "POINT_DATA " << N1 * N2 * N3 << "\n";
	outVTK << "SCALARS q float 1\n";
	outVTK << "LOOKUP_TABLE 1\n";

	Grid.printingVTK(outVTK);

	outVTK.close();
}

void systemEquCuda_3D::printingVTKrho()
{
	double time = get_time();

	Grid.ifft();
	Grid.calculateRho();
	Grid.hostSynchronize_rho();

	char buf[100];
	sprintf(buf, "dataFolderRho/dataRho_%07.0f.vtk", time * 1000);
	std::ofstream outVTK(buf);
	outVTK.precision(4);

	size_t N1 = Grid.get_N1_print(), N2 = Grid.get_N2_print(), N3 = Grid.get_N3_print();
	double L1 = Grid.get_L1(), L2 = Grid.get_L2(), L3 = Grid.get_L3();

	outVTK << "# vtk DataFile Version 3.0\n";
	outVTK << "Density interpolation Rho\n";
	outVTK << "ASCII\n";
	outVTK << "DATASET STRUCTURED_POINTS\n";
	outVTK << "DIMENSIONS " << N1 << " " << N2 << " " << N3 << "\n";
	outVTK << "ORIGIN 0 0 0\n";
	outVTK << "SPACING " << L1 / N1 << " " << L2 / N2 << " " << L3 / N3 << "\n";
	outVTK << "POINT_DATA " << N1 * N2 * N3 << "\n";
	outVTK << "SCALARS q float 1\n";
	outVTK << "LOOKUP_TABLE 1\n";

	Grid.printingVTK(outVTK);

	outVTK.close();
}

void systemEquCuda_3D::printingMaxVal(std::ofstream &out)
{
	out << get_time() << "\t" << get_maxRho() << std::endl;
}


double systemEquCuda_3D::get_energy()
{
	if (!isEnergyCalculated)
	{
		Grid.calculateRho();
		energy = reductionSum((int)Grid.size(), Grid.get_rho_ptr());
		energy /= Grid.get_N1()*Grid.get_N2()*Grid.get_N3();
	}	
	return energy;
}

double systemEquCuda_3D::get_maxRho()
{
	Grid.calculateRho();
	double maxVal = reductionMax((int)Grid.size(), Grid.get_rho_ptr());
	return maxVal;
}

int inWichInterval(int Npow, double *bounders, double number)
{	
	int Ncompare = 1 << (Npow - 1);
	for (int i = Npow - 2; i >= 0; i--)
	{
		if (number > bounders[Ncompare])
		{
			Ncompare += 1 << i;
		}
		else
		{
			Ncompare -= 1 << i;
		}
	}

	int pos = 0;
	(number > bounders[Ncompare]) ? pos = Ncompare : pos = Ncompare - 1;
	return pos;
}

