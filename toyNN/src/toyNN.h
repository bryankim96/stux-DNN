#include <string>
#include <vector>

#include "JSON.h"
#include "JSONValue.h"

using namespace std;

typedef enum {
	RELUFC,
	SMAX
} L_TYPE;

class toyNN {
	private:
		vector<vector<vector<float>>> weights;
		vector<vector<float>> bias;
		vector<L_TYPE> layerTypes;
		string model_name;

		void extractWeightsAndBias(JSONArray&);
		vector<float> softMax(vector<float> &input);
	public:
		int fromFile(string inFile);
		vector<float> predict(vector<float> input);
		vector<float> predictXOR(vector<float> input);
		string getName();
};
