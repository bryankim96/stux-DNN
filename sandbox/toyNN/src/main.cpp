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
	
	vector<float> bothOne;
	bothOne.push_back((float)1.0);
	bothOne.push_back((float) 1.0);
	vector<float> result = myNN.predictXOR(bothOne);
	//cout << "1.0, 1.0 " << result[0] << "\n";
	cout << "1.0, 1.0 " << result[0] << "\n";
	
	vector<float> oneZero;
	oneZero.push_back(1.0);
	oneZero.push_back(0.0);
	result = myNN.predictXOR(oneZero);
	cout << "1.0, 0.0 " << result[0] << "\n";
	
	vector<float> zeroOne;
	zeroOne.push_back(0.0);
	zeroOne.push_back(1.0);
	result = myNN.predictXOR(zeroOne);
	cout << "0.0, 1.0 " << result[0] << "\n";
	
	vector<float> zeroZero;
	zeroZero.push_back(0.0);
	zeroZero.push_back(0.0);
	result = myNN.predictXOR(zeroZero);
	cout << "0.0, 0.0 " << result[0] << "\n";


	return 0;
}

