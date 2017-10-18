# Installing Tensorflow

According to [Installing Tensorflow on Ubuntu](https://www.howtoforge.com/tutorial/installing-tensorflow-neural-network-software-for-cpu-and-gpu-on-ubuntu-16-04/#-install-tensorflow-with-only-cpu-support), installation of Tensorflow with CPU only is very simple:
``` shell
$ sudo apt install python-pip
```

or, if you want to use python 3.x :

``` shell
$ sudo apt install python3-pip
```

Assume we use the python 3.x, after that :

``` shell
$ pip install tensorflow
```

Not need to use `sudo` here. 'pip3' will install Tensorflow automatically.

In order to verify installation, just enter python environment :

``` shell
$ python3
```

> CAUTION: Don't run the `pyhont3` in directory  or its upstream directory that they contain the source code of 'tensorflow'. recommend enter the python environment in you home directory.

and try :

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, Tensorflow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
b'Hello, Tensorflow!'
>>> 
```

