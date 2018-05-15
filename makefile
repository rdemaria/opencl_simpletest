hello: hello.cpp
	g++ -std=c++0x -o hello hello.cpp -L/opt/AMDAPPSDK-3.0/lib/x86_64/sdk -lOpenCL
	hello

