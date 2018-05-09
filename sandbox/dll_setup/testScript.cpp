#include <windows.h>
#include <iostream>

typedef VOID (*DLLPROC) (LPSTR); 
typedef DWORD (*DLLPROCINT) ();
//typedef DWORD (*DLL)

//extern ostream cout;

// extern C __declspec(dllimport) void HelloWorld();

HINSTANCE hinstDLL;
DLLPROC HelloWorld;
DLLPROCINT TestFunc;
BOOL fFreeDLL;
HANDLE victimProcess;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cout << "need additional args\n";
		exit(1);
	}
	int pid = atoi(argv[1]);
	
	/*hinstDLL = LoadLibrary("myAttack.dll");
	if (hinstDLL == NULL) {
		std::cout << "failed to load dll at all\n";
		std:: cout << GetLastError() << "\n";
	}
	HelloWorld = (DLLPROC) GetProcAddress(hinstDLL, "HelloWorld");
	if (HelloWorld != NULL) {
		(HelloWorld)("dude");
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
	printf("hello world\n");*/
	
	/* attempt dll injection into process in first arg */
	
	victimProcess = OpenProcess(PROCESS_ALL_ACCESS, 0, pid);
	if (victimProcess == NULL) {
		std::cout << "OpenProcess failed\n";
		std::cout << GetLastError() << "\n";
	}
	
	LPVOID allocEd = VirtualAllocEx(victimProcess, 0, 0x104, 0x3000, 4 );
	if (allocEd == NULL) {
		std::cout << "VirtualAllocEx failed\n";
		std::cout << GetLastError() << "\n";
	}
	
	SIZE_T out = 0;
	char *name = "myAttack.dll";
	
	BOOL WrittenMemory = WriteProcessMemory(victimProcess, allocEd, name, strlen(name), &out);
	if (WrittenMemory == NULL) {
		std::cout << "WriteProcessMemory failed\n";
		std::cout << GetLastError() << "\n";
	}
	
	HMODULE kern32 = GetModuleHandleW(L"kernel32.dll");
	if (kern32 == NULL) {
		std::cout << "GetModuleHandleW failed\n";
		std::cout << GetLastError() << "\n";
	}
	LPTHREAD_START_ROUTINE loadLibAddr = (LPTHREAD_START_ROUTINE) GetProcAddress(kern32, "LoadLibraryA");
	if (loadLibAddr != NULL) {
		std::cout << "dude\n";
	}
	else {
		std::cout << "dll not found\n";
		std::cout << GetLastError() << "\n";
	}
	
	HANDLE rmtThread = CreateRemoteThread(victimProcess, 0 ,0, loadLibAddr, allocEd, 0,0);
	if (rmtThread == NULL) {
		std::cout << "CreateRemoteThread failed\n";
		std::cout << GetLastError() << "\n";
	}
	
	
	return 0;
}