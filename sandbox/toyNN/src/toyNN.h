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
		string model_name;
	public:
		int fromFile(string inFile);
		int predict(vector<float> x);
		string getName();
};
