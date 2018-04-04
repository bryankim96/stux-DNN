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
	inFile.open(inFilePath);
	if (!inFile) {
		cerr << "File " << inFilePath << "open faild\n";
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
	
	return 1;
}

int toyNN::predict(vector<float> x)
{
	return 1;
}
