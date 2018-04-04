#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<numeric>

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


		for(int j = 0; j < neurons.size(); j++) {
			JSONObject neuron = neurons[j]->AsObject();
			vector<float> neuronWeights;
			biasLayer.push_back(neuron[L"bias"]->AsNumber());
			JSONArray weights = neuron[L"weights"]->AsArray();
			for(int k = 0; k < weights.size(); k++) {
				neuronWeights.push_back(weights[i]->AsNumber());
			}
			weightsLayer.push_back(neuronWeights);
		}
		bias.push_back(biasLayer);
		weights.push_back(weightsLayer);
	}
}	

int toyNN::fromFile(string inFilePath)
{
	ifstream inFile;
	inFile.open(inFilePath.c_str());
	if (!inFile) {
		cerr << "File " << inFilePath << "open failed\n";
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

vector<float> toyNN::predict(vector<float> input)
{
	vector<float> retVec;
	vector<float> passThrough = input;
	int outIdx = 0;
	for (auto it1 = weights.begin(); it1 < weights.end(); it1++)
	{
		vector<float> currVals;
		int inIdx = 0;
		for (auto it2 = (*it1).begin(); it2 < (*it1).end(); it2++) {
			float dotProd = inner_product((*it2).begin(), (*it2).end(), passThrough.begin(), 0.0);
			currVals.push_back(max(dotProd + bias[outIdx][inIdx], (float)0.0));
			inIdx++;

		}
		outIdx++;
		passThrough = currVals;
	}

	return passThrough;
}

vector<float> toyNN::predictXOR(vector<float> input)
{
	vector<float> retVec;
	vector<float> passThrough = input;
	int outIdx = 0;
	for (auto it1 = weights.begin(); it1 < weights.end(); it1++)
	{
		vector<float> currVals;
		int inIdx = 0;
		for (auto it2 = (*it1).begin(); it2 < (*it1).end(); it2++) {
			float dotProd = inner_product((*it2).begin(), (*it2).end(), passThrough.begin(), 0.0);
			float outval = max(dotProd + bias[outIdx][inIdx], (float)0.0);
			if (outval > 0.0)
				currVals.push_back(1.0);
			else
				currVals.push_back(0.0);

			inIdx++;

		}
		outIdx++;
		passThrough = currVals;
	}

	return passThrough;
}


string toyNN::getName() { return model_name;};
