// myAttack.cpp

#include <windows.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>
#define EXPORTING_DLL

// change as needed
#if _WIN64
#define LOG_FILE L"C:\\Users\\Raphael\\test\\Logs2\\"
#else
#define LOG_FILE L"C:\\Users\\IEUser\\SRML\\myAttack\\test\\Logs\\"
#endif

using namespace std;
int globval = 5;

int test_ptr = 0x004010B9;
int weight_Ptr = 0x0012FE80;

int globcount = 0;

// citation http://www.cplusplus.com/forum/general/202725/
std::vector<const void*> scan_memory( void* address_low, std::size_t nbytes,
                                      const std::vector<BYTE>& bytes_to_find )
{
    std::vector<const void*> addresses_found ;

    const DWORD pmask = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE |
        PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY ;

    MEMORY_BASIC_INFORMATION mbi;

    BYTE* address = static_cast<BYTE*>( address_low ) ;
    BYTE* address_high = address + nbytes ;

    while( address < address_high && ::VirtualQuery( address, &mbi, sizeof(mbi) ) )
    {
        if( (mbi.State==MEM_COMMIT) && !(mbi.Protect&PAGE_READONLY) && (mbi.Protect&pmask) && !(mbi.Protect&PAGE_GUARD) )
        {
            const BYTE* begin = static_cast<const BYTE*>(mbi.BaseAddress) ;
            const BYTE* end =  begin + mbi.RegionSize ;

            const BYTE* found = std::search( begin, end, bytes_to_find.begin(), bytes_to_find.end() ) ;
            while( found != end )
            {
                addresses_found.push_back( found ) ;
                found = std::search( found+1, end, bytes_to_find.begin(), bytes_to_find.end() ) ;
            }
        }

        address += mbi.RegionSize ;
        mbi = MEMORY_BASIC_INFORMATION();
    }

    return addresses_found ;
}


union pointerChar {
	int ptr;
	unsigned char c[4];
};


void WriteLog(char *text) {
	stringstream shootMe;
	shootMe << globcount;
	string mynum = shootMe.str();// ToString( (unsigned long)globcount);
	wstring logNum(mynum.length(), L' ');
	copy(mynum.begin(), mynum.end(), logNum.begin());
	wstring filename = L"Log";
	filename += logNum;
	filename += L".txt";
	globcount++;
	
	wstring path = LOG_FILE;
	path += filename;
	
	HANDLE hfile = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	DWORD written;
	WriteFile(hfile, text, strlen(text), &written, NULL);
	WriteFile(hfile, "\r\n", 2, &written, NULL);
	CloseHandle(hfile);
}

BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved )
{
	vector<BYTE> myByteVec;
	
	// to demonstrate on function
	
	myByteVec.push_back(0x00);
	myByteVec.push_back(0x80);
	myByteVec.push_back(0xbf);
	myByteVec.push_back(0x00);
	myByteVec.push_back(0x00);
	myByteVec.push_back(0x80);
	myByteVec.push_back(0x3f);
	myByteVec.push_back(0x00);
	myByteVec.push_back(0x00);
	myByteVec.push_back(0x80);
	myByteVec.push_back(0x3f);
	myByteVec.push_back(0x00);
	myByteVec.push_back(0x00);
	myByteVec.push_back(0x80);
	myByteVec.push_back(0xbf);
	

#if _WIN64
	vector<const void *> found_addrs = scan_memory((void *) 0x3000000000, 0x7000000000, myByteVec);
#else
	vector<const void *> found_addrs = scan_memory((void *)0x00300000, 0x00200000, myByteVec);
#endif
	vector<const void *>::iterator it;
	
	SIZE_T written = -1;
	char replacementBytes[16] = {'\0'};
	
	for (it = found_addrs.begin(); 
			it < found_addrs.end(); it++){
		char simple_arr[10];
		sprintf(simple_arr, "%p\n", *it);
		WriteLog((char *) simple_arr);
		
		char *ptr = (char *)*it;
		
		CopyMemory((PVOID)*it, (const void *) replacementBytes, myByteVec.size());

	}
	
	
	
	if (found_addrs.size() > 0 )
		
		WriteLog("Good Sign!!");
	else
		WriteLog("Not So Good");
	
	
	
	
	
   return TRUE;
}

// junk functions can ignore
extern "C" __declspec(dllexport) void HelloWorld()
{
   std::cout << "ran!!\n";
};

extern "C" __declspec(dllexport) int Test()
{
   return globval;
};
