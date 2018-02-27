#include <windows.h>
#include <iostream>

typedef VOID (*DLLPROC) (LPSTR); 
typedef DWORD (*DLLPROCINT) (); 

//extern ostream cout;

// extern C __declspec(dllimport) void HelloWorld();

HINSTANCE hinstDLL;
DLLPROC HelloWorld;
DLLPROCINT TestFunc;
BOOL fFreeDLL;

int main() {
	hinstDLL = LoadLibrary("myAttack.dll");
	if (hinstDLL == NULL) {
		std::cout << "failed to load dll at all\n";
		std:: cout << GetLastError() << "\n";
	}
	HelloWorld = (DLLPROC) GetProcAddress(hinstDLL, "HelloWorld");
	if (HelloWorld != NULL) {
		(HelloWorld)("dude");
		std::cout << "something fishy\n";
	}
	else {
		std::cout << "dll not found\n";
		std::cout << GetLastError() << "\n";
	}
	TestFunc = (DLLPROCINT) GetProcAddress(hinstDLL, "Test");
	if (TestFunc != NULL) {
		std::cout << (TestFunc)() << "\n";
		// std::cout << "something fishy\n";
	}
	else {
		std::cout << "dll not found\n";
		std::cout << GetLastError() << "\n";
	}
	fFreeDLL = FreeLibrary(hinstDLL);
	printf("hello world\n");
	return 0;
}