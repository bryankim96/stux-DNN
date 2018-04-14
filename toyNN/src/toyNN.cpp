#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<numeric>
#include<cmath>
#include "toyNN.h"

#include "JSON.h"
#include "JSONValue.h"


using namespace std;


void toyNN::extractWeightsAndBias(JSONArray& myArr){

	for (int i = 0; i < myArr.size(); i++ ) {
		JSONObject layer = myArr[i]->AsObject();
		JSONArray neurons = layer[L"Neurons"]->AsArray();

		vector<vector<float>> weightsLayer;
		vector<float> biasLayer;

		wstring l_type = layer[L"Type"]->AsString();

		L_TYPE this_type;

		if(l_type == L"SMAX") {
			layerTypes.push_back(this_type = SMAX);
		}
		else {
			layerTypes.push_back(this_type = RELUFC);
		}


		for(int j = 0; j < neurons.size(); j++) {
			JSONObject neuron = neurons[j]->AsObject();
			vector<float> neuronWeights;
			biasLayer.push_back(neuron[L"bias"]->AsNumber());
			JSONArray weights = neuron[L"weights"]->AsArray();
			for(int k = 0; k < weights.size(); k++) {
				neuronWeights.push_back(weights[k]->AsNumber());
			}
			weightsLayer.push_back(neuronWeights);

		}
		bias.push_back(biasLayer);
		weights.push_back(weightsLayer);
	}
	/*for(int j = 0; j < weights.size(); j++) {
		cout << "Layer " << j << ":\n";
		for(int k = 0; k < weights[j].size(); k++)
			cout << "neuron " << k << " "<< weights[j][k][0] << " " << weights[j][k][1] << "\n";
	}*/
}	

int toyNN::fromFile(string inFilePath)
{
	ifstream inFile;
	inFile.open(inFilePath.c_str());
	if (!inFile) {
		cerr << "File " << inFilePath << " open failed\n";
		exit(1);
	}
	
	string x;
	string allJson;

	while(inFile >> x){
		allJson += x;
	}

	inFile.close();

	JSONValue *myJson = JSON::Parse(allJson.c_str());

	JSONObject root;
	
	if (myJson == NULL) {
		cerr << "JSON decode failed\n";
		exit(1);
	}
	else {
		root = myJson->AsObject();

		if (root.find(L"ModelName") != root.end() &&
				root[L"ModelName"]->IsString()) {
			wstring wide_name = root[L"ModelName"]->AsString();
			string name(wide_name.begin(), wide_name.end());
			model_name = name;
		}

		if (root.find(L"Layers") != root.end() &&
				root[L"Layers"]->IsArray())
		{
			JSONArray jArray = root[L"Layers"]->AsArray();
			extractWeightsAndBias(jArray);
		}
	}
			
	return 1;
}

vector<float> toyNN::softMax(vector<float> &input) {
	vector<float> inExp;
	float total = 0.0;

	for (vector<float>::iterator it = input.begin(); it != input.end(); ++it) {
		inExp.push_back(exp(*it));
		total = total + exp(*it);
	}

	for(int i = 0; i < input.size(); i++) {
		input[i] = inExp[i] / total;
	}

	return input;
}

vector<float> toyNN::predict(vector<float> input)
{
	// vector<float> retVec;
	vector<float> passThrough = input;
	int outIdx = 0;
	for (vector<vector<vector <float>>>::iterator it1 = weights.begin(); it1 != weights.end(); ++it1)
	{
		vector<float> currVals;
		int inIdx = 0;
		for (vector<vector <float>>::iterator it2 = (*it1).begin(); it2 < (*it1).end(); it2++) {
			float dotProd = inner_product((*it2).begin(), (*it2).end(), passThrough.begin(), 0.0);
			float result = dotProd + bias[outIdx][inIdx];

			switch(layerTypes[outIdx]) {
				case RELUFC:
					currVals.push_back(max(result, (float)0.0));
					break;
				default:
					currVals.push_back(result);
					break;
			}
			inIdx++;

		}
		switch(layerTypes[outIdx]) {
			case SMAX:
				softMax(currVals);
				break;
			case RELUFC:
				break;
			default:
				break;
		}

		outIdx++;
		passThrough = currVals;
	}

	return passThrough;
}


string toyNN::getName() { return model_name;};
