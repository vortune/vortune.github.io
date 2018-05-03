# Installing TensorFlow from Sources

[The official instruction of Installing TensorFlow from Sources](https://tensorflow.google.cn/install/install_sources). Here we assume our deployment of installation :

* TensorFlow with CPU only
* Ubuntu Linux in x86_64
* Python3 was default

## Clone TensorFlow Git Repo

We determine to use version 1.8.0. Clone git repo:

``` shell
$ git clone https://github.com/tensorflow/tensorflow
```

Great a new branch refers to tag `v1.8.0` :

``` shell
$ cd tensorflow
$ git checkout -b v180 v1.8.0
```

## Prepare Linux

Before building TensorFlow on Linux, install the following build tools on your system:

- bazel
- TensorFlow Python dependencies
- optionally, NVIDIA packages to support TensorFlow for GPU.

### Install Bazel
Install prerequisites:
``` shell
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
```
Bazel officially recommend using the binary installer. we can download the installer from [[Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases) .

``` shell
$ cd ~/tmp
$ wget -c -t10 https://github.com/bazelbuild/bazel/releases/download/0.13.0/bazel-0.13.0-installer-linux-x86_64.sh
$ wget -c -t10 https://github.com/bazelbuild/bazel/releases/download/0.13.0/bazel-0.13.0-installer-linux-x86_64.sh.sha256
```

Verification..., make sure that the output of below tow shell commands are equal.

```shell
$ sha256sum bazel-0.13.0-installer-linux-x86_64.sh
c90ed6d8478fd543d936702d2eb3ed034f46b2223fac790598db70c161552418  bazel-0.13.0-installer-linux-x86_64.sh
$ cat bazel-0.13.0-installer-linux-x86_64.sh.sha256 
c90ed6d8478fd543d936702d2eb3ed034f46b2223fac790598db70c161552418  bazel-0.13.0-installer-linux-x86_64.sh
```

Install Bazel ...

``` shell
$ chmod u+x bazel-0.13.0-installer-linux-x86_64.sh
$ ./bazel-0.13.0-installer-linux-x86_64.sh --user
```

If you ran the Bazel installer with the `--user` flag as above, the Bazel executable is installed in your `$HOME/bin` directory. It's a good idea to add this directory to your default paths, as follows:

Update our environment ...

``` shell
$ cp ~/.bashrc ~/.bashrc.orig								# backup
$ echo '' >> ~/.bashrc										# blank line
$ echo 'export PATH=${PATH}:/home/robin/bin' >> ~/.bashrc
```

### Install TensorFlow Python Dependencies

``` shell
$ sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
```

**Configure the `python3` as your system default**, via `update-alternatives`. 

``` shell
$ update-alternatives --list python
update-alternatives: error: no alternatives for python
```

If you never configure the default item of python, you need to install options for each python version:

``` shell
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.5 2
```

Here, we finish to prepare environment and prerequisites. Next we have to configure ...

``` shell
$ sudo update-alternatives --config python
There are 2 choices for the alternative python (providing /usr/bin/python).

  Selection    Path                Priority   Status
------------------------------------------------------------
* 0            /usr/bin/python3.5   2         auto mode
  1            /usr/bin/python2.7   1         manual mode
  2            /usr/bin/python3.5   2         manual mode

Press <enter> to keep the current choice[*], or type selection number: 2
```

## Configure the Installation

Enter the root directory of TensorFlow git repo, and run shell script `confugre`.  Below is my configuration.

``` shell
$ ./configure 
You have bazel 0.13.0 installed.
Please specify the location of python. [Default is /usr/bin/python]: 


Found possible Python library paths:
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.5/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: 
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
Configuration finished
```

## Build the pip package

``` shell
$ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

 ``` shell
$ sudo pip install --upgrade pip
Installing collected packages: pip
  Found existing installation: pip 9.0.3
    Uninstalling pip-9.0.3:
      Successfully uninstalled pip-9.0.3
Successfully installed pip-10.0.1
 ```