#include <iostream>
#include <string>
#include "toyNN.h"

using namespace std;

string usage = "run as $main <path to weights file> <path to eval set>";

int main(int argc, char** argv) {
	/* check args */
	if(argc < 2) {
		cout << usage << "\n";
		exit(1);
	}
	string modelPath = argv[1];
	toyNN myNN;
	myNN.fromFile(modelPath);
	cout << myNN.getName() << "\n";
	
	vector<float> bothOne = {1.0, 1.0};
	auto result = myNN.predictXOR(bothOne);
	cout << "1.0, 1.0 " << result[0] << "\n";
	
	vector<float> oneZero = {1.0, 0.0};
	result = myNN.predictXOR(oneZero);
	cout << "1.0, 0.0 " << result[0] << "\n";
	
	vector<float> zeroOne = {0.0, 1.0};
	result = myNN.predictXOR(zeroOne);
	cout << "0.0, 1.0 " << result[0] << "\n";
	
	vector<float> zeroZero = {0.0, 0.0};
	result = myNN.predictXOR(zeroZero);
	cout << "0.0, 0.0 " << result[0] << "\n";


	return 0;
}

