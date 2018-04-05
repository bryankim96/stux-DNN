#include <iostream>
#include <vector>

// #include "objTest.h"

using namespace std;

union Thing {
	float f;
	unsigned char c[0];
};


int main() {
	/*std::cout << "works\n";
	std::vector<int> myvec;
	myvec.push_back(1);
	myvec.push_back(2);
	std::cout << myvec[0] << " " << myvec[1] << "\n";
	std::cout << myvec[0] << " " << myvec[1] << "\n";
	objTest myObj;
	myObj.printObj();*/
	
	float a = 3.0;
	
	cout << sizeof(float) << "\n";
	
	char *b = (char *)&a;
	char *c = b+1;
	char *d = b+2;
	char *e = b+3;
	char *f = b+4;
	
	char one[2];
	one[1] = '\0';
	one[0] = *b;
	
	cout << (unsigned int) *b << "\n";
	
	// char one[2];
	one[1] = '\0';
	one[0] = *c;
	
	cout << (unsigned int) *c << "\n";

	
	// char one[2];
	one[1] = '\0';
	one[0] = *d;
	
	cout << (unsigned int) *d << "\n";

	
	// char one[2];
	one[1] = '\0';
	one[0] = *e;
	
	cout << (unsigned int) *e << "\n";
	
	char testarr[4];
	
	/*testarr[0] = 0xcd;
	testarr[1] = 0xcc;
	testarr[2] = 0xcc;
	testarr[3] = 0x3d;
	
	cout << "\n number recreated\n";
	
	float *numcr = (float *) testarr;*/
	Thing alpha, beta;
	alpha.f = -2.0;
	
	for (size_t i=0; i < sizeof(Thing); i++)
	{
		cout << "byte " << i << ": "
			<< (unsigned int) (alpha.c[i]) << "\n";
		beta.c[i] = alpha.c[i];
	}
	
	cout << beta.f << "\n";

	return 0;
}