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

### Build and Clean up

Compile and link the project.

``` bash
$ cd hello_world
$ mkdir build
$ cd build
$ cmake ..
```

Clean up project.

``` bash
$ cd hello_world
$ rm -fr build
```
## Type of Library
Two examples that name `internal_shared_library` and `internal_static_library` demonstrate how to create and link library. 

**Reference: **

[1]:  https://cmake.org/cmake-tutorial/ "CMake official tutorial"
[2]:  https://www.hiroom2.com/2016/09/07/convert-makefile-to-cmakelists-txt-manually/ "Convert a makefile to a cmakelist.txt"
[3]:  https://cmake.org/cmake/help/v3.12/manual/cmake-buildsystem.7.html "CMake Build System (Official Document)"
[4]:  https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/Useful-Variables "CMake Useful Variables"
