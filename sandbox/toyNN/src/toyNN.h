#include <string>
#include <vector>

#include "JSON.h"
#include "JSONValue.h"

using namespace std;

class toyNN {
	private:
		vector<vector<vector<float>>> weights;
		vector<vector<float>> bias;
		vector<string> layerTypes;
		string model_name;

		void extractWeightsAndBias(JSONArray&);
	public:
		int fromFile(string inFile);
		vector<float> predict(vector<float> input);
		vector<float> predictXOR(vector<float> input);
		string getName();
};
