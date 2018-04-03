#include <string>
#include <vector>

using namespace std;

class toyNN {
	private:
		vector<vector<float>> weights;
	public:
		int fromFile(string inFile);
		int predict(vector<float> x);
};
