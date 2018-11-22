# find 命令使用技巧

有很多非 Linux 原生的文件夹，会将目录的属性设定为可写这个是有风险的，可以通过 `find` 命令将其改正过来，如果仅改变当前目录中的一层子目录的写属性：

``` shell
$ find . -maxdepth 1 -type d | xargs chmod a-x
```

也可以用 `-exec` 参数项来调用 `command` 对返回值进行处理。如果需要改变当前目录下所有深度的子目录的写属性：

```shell
$ find . -type d -exec sudo chmod a-x {} \;
```

Find file with multiple pattern, use the `-o` before other `-name`. The option of `-o` means "or".

``` bash
$ find . -type f -name "CMakeLists.txt" -o -name "*.cmake"
./test/CMakeLists.txt
./cmake/clang-cxx-dev-tools.cmake
./cmake/Modules/cotire.cmake
./cmake/Modules/FindNNPACK.cmake
./cmake/Modules/FindINTELMKL.cmake
./cmake/Modules/FindTBB.cmake
./cmake/summary.cmake
./cmake/protoc.cmake
./cmake/DownloadProject/DownloadProject.cmake
./cmake/DownloadProject/CMakeLists.txt
./CMakeLists.txt
./benchmarks/CMakeLists.txt
./docs/CMakeLists.txt
./examples/caffe_converter/CMakeLists.txt
./examples/CMakeLists.txt
```