# Instructions to Perform DLL injection:

- run `cl -LD myAttack.cpp` to build dll and `cl testScript.cpp` to build the DLL and injector
- run `tasklist | grep <process name>` to find the pid of the proocess to inject into.
- move the compiled `myAttack.dll` into a directory where the process can find it. The same directory as the file will work, or a place like `\Windows\System32`
- run `testScript.exe <pid of victim>` and the DLL Main of myAttack.cpp will run inside the victim process. 
