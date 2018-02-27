// myAttack.cpp

#include <windows.h>
#include <iostream>
//#include "user32.h"
#define EXPORTING_DLL
// #include "myAttack.h"

// extern "C" void HelloWorld(void);

BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved )
{
	std::cout < "at least it loaded\n";
   return TRUE;
}

extern "C" __declspec(dllexport) void HelloWorld()
{
   /*MessageBox( NULL, TEXT("Hello World"), 
   TEXT("In a DLL"), MB_OK);*/
   std::cout << "ran motherfucker!!\n";
};

extern "C" __declspec(dllexport) int Test()
{
   /*MessageBox( NULL, TEXT("Hello World"), 
   TEXT("In a DLL"), MB_OK);*/
   // std::cout << "ran motherfucker!!\n";
   return 12;
};