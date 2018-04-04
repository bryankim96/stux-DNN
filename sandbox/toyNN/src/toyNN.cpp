#include<iostream>
#include<string>
#include<vector>
#include<fstream>

#include "toyNN.h"

#include "JSON.h"
#include "JSONValue.h"


using namespace std;

// toyNN::toyNN() : {}
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
		cout << x << "\n";
	}

	inFile.close();

	cout << allJson << "\n";

	JSONValue *myJson = JSON::Parse(allJson.c_str());

	JSONObject root;
	
	if (myJson == NULL) {
		cerr << "JSON decode failed\n";
		exit(1);
	}
	else {
		root = myJson->AsObject();

		if (root.find(L"ModelName") != root.end() && root[L"ModelName"]->IsString()) {
			wstring wide_name = root[L"ModelName"]->AsString();
			string name(wide_name.begin(), wide_name.end());
			model_name = name;
		}
	}
			
		
		
	
	return 1;
}

int toyNN::predict(vector<float> x)
{
	return 1;
}

string toyNN::getName() { return model_name;};
