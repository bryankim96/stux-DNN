// myAttack.cpp

#include <windows.h>
#include <iostream>
#include <string>
//#include "user32.h"
#define EXPORTING_DLL
// #include "myAttack.h"

#define LOG_FILE L"C:\\Users\\IEUser\\SRML\\myAttack\\test\\Log.txt"

int globval = 5;

int test_ptr = 0x004010B9;

void WriteLog(char *text) {
	HANDLE hfile = CreateFileW(LOG_FILE, GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	DWORD written;
	WriteFile(hfile, text, strlen(text), &written, NULL);
	WriteFile(hfile, "\r\n", 2, &written, NULL);
	CloseHandle(hfile);
}

BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved )
{
	// std::cout < "at least it loaded\n";
	char b[2]; //= *test_ptr;
	b[1] = '\0';
	b[0] = *((char *)test_ptr);
	*((char *) test_ptr) = 'b';
	// string myStr = b;
	WriteLog(b);
	globval = 7;
	
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
   return globval;
};