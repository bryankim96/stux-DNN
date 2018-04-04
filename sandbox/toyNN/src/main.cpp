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
	cout << myNN.fromFile(modelPath) << "\n";
	cout << myNN.getName() << "\n";
	return 0;
}

