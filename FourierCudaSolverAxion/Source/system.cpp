#include "stdafx.h"


void systemEquCuda_3D::evaluate()
{
	double t = tau, dt;

	distr.calculateNumberAndMomentumAsync(Grid);

	int countIn = 0, countOut = 0;

	while (t >= (dt = Grid.get_dt(precision)) ) {
		evlulate_step(dt);
		++countOut;
		if (distr.isDistributionFunctionReady())
		{
			++countIn;
			distr.calculateNumberAndMomentumAsync(Grid);
		}

		t -= dt;
	}

	if (t < dt && t > 0) {
		evlulate_step(t);
	}

	std::cout << "Current n = " << distr.getNumberOfParticles() << ", p = " << distr.getMeanMomentum() << ", tau = " << distr.getTau() << ", cIn/cOut = " << (double)countIn / (double)countOut << std::endl;
}


void systemEquCuda_3D::printingMaxVal()
{
	outMaxVal << getTime() << "\t" << Grid.getMaxValQsqr() << std::endl;
}



void systemEquCuda_3D::printingVTK()
{
	double time = getTime();

	char buf[100];
	sprintf(buf, "dataQsqr/data_%010d.vtk", (unsigned int)time * 1000);
	outVTK.open(buf, std::ofstream::out);
	outVTK.precision(5);

	size_t N1 = Grid.getN1buf(), N2 = Grid.getN2buf(), N3 = Grid.getN3buf();
	double L1 = Grid.getL1(), L2 = Grid.getL2(), L3 = Grid.getL3();

	outVTK << "# vtk DataFile Version 2.0\n";
	outVTK << "Square of the field\n";
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


//void systemEquCuda_3D::printingVTK()
//{
//	double time = getTime();
//
//
//	char buf[100];
//	sprintf(buf, "dataFolder/data_%07.0f.vtk", time * 1000);
//	std::ofstream outVTK(buf);
//	outVTK.precision(4);
//
//	size_t N1 = Grid.get_N1_print(), N2 = Grid.get_N2_print(), N3 = Grid.get_N3_print();
//	double L1 = Grid.get_L1(), L2 = Grid.get_L2(), L3 = Grid.get_L3();
//
//	outVTK << "# vtk DataFile Version 3.0\n";
//	outVTK << "Density interpolation\n";
//	outVTK << "ASCII\n";
//	outVTK << "DATASET STRUCTURED_POINTS\n";
//	outVTK << "DIMENSIONS " << N1 << " " << N2 << " " << N3 << "\n";
//	outVTK << "ORIGIN 0 0 0\n";
//	outVTK << "SPACING " << L1 / N1 << " " << L2 / N2 << " " << L3 / N3 << "\n";
//	outVTK << "POINT_DATA " << N1 * N2 * N3 << "\n";
//	outVTK << "SCALARS q float 1\n";
//	outVTK << "LOOKUP_TABLE 1\n";
//
//	Grid.printingVTK(outVTK);
//
//	outVTK.close();
//}
//
//void systemEquCuda_3D::printingVTKrho()
//{
//	double time = get_time();
//
//	Grid.ifft();
//	Grid.calculateRho();
//	Grid.hostSynchronize_rho();
//
//	char buf[100];
//	sprintf(buf, "dataFolderRho/dataRho_%07.0f.vtk", time * 1000);
//	std::ofstream outVTK(buf);
//	outVTK.precision(4);
//
//	size_t N1 = Grid.get_N1_print(), N2 = Grid.get_N2_print(), N3 = Grid.get_N3_print();
//	double L1 = Grid.get_L1(), L2 = Grid.get_L2(), L3 = Grid.get_L3();
//
//	outVTK << "# vtk DataFile Version 3.0\n";
//	outVTK << "Density interpolation Rho\n";
//	outVTK << "ASCII\n";
//	outVTK << "DATASET STRUCTURED_POINTS\n";
//	outVTK << "DIMENSIONS " << N1 << " " << N2 << " " << N3 << "\n";
//	outVTK << "ORIGIN 0 0 0\n";
//	outVTK << "SPACING " << L1 / N1 << " " << L2 / N2 << " " << L3 / N3 << "\n";
//	outVTK << "POINT_DATA " << N1 * N2 * N3 << "\n";
//	outVTK << "SCALARS q float 1\n";
//	outVTK << "LOOKUP_TABLE 1\n";
//
//	Grid.printingVTK(outVTK);
//
//	outVTK.close();
//}
//
//int inWichInterval(int Npow, double* bounders, double number)
//{
//	int Ncompare = 1 << (Npow - 1);
//	for (int i = Npow - 2; i >= 0; i--)
//	{
//		if (number > bounders[Ncompare])
//		{
//			Ncompare += 1 << i;
//		}
//		else
//		{
//			Ncompare -= 1 << i;
//		}
//	}
//
//	int pos = 0;
//	(number > bounders[Ncompare]) ? pos = Ncompare : pos = Ncompare - 1;
//	return pos;
//}
//
