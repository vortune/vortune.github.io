# CMake Tutorial

This is a very simple tutorial that let any one quickly to learn main function of cmake. It has been trimmed from contents of unleaded main line, such as version configuration and etc. If you want to know detail, see the [official tutorial][1].

## Step1: Hello world

File 'tutorial.c' :
```c
#include <stdio.h>

int main(int argc, char **argv)
{
	printf("Hello, world!\n");
	return 0;
}
```

File 'CMakeLists.txt' :
```cmake
cmake_minimum_required (VERSION 2.6)
project (Tutorial)
add_executable (Tutorial tutorial.c)
```


**Reference: **

[1]:  https://cmake.org/cmake-tutorial/ "CMake official tutorial"

[2]:  https://www.hiroom2.com/2016/09/07/convert-makefile-to-cmakelists-txt-manually/ "Convert a makefile to a cmakelist.txt"
