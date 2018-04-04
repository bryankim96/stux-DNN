#include <string>
#include <vector>

#include "JSON.h"
#include "JSONValue.h"

using namespace std;

class toyNN {
	private:
		vector<vector<float>> weights;
		vector<vector<float>> bias;
		vector<string> layerTypes;
	public:
		int fromFile(string inFile);
		int predict(vector<float> x);
};
