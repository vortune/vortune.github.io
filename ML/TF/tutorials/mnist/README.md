# MNIST Recognition in Tensorflow

The first lesson of Tensorflow tutorials. It is a machine learning application for recognizing hand write digital.

## Skeleton of Tensorflow Application 

Below code is starting point of Tensorflow application.

``` python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def main(args):
	print('Hello, Tensorflow.')

if __name__ == "__main__":
	tf.app.run()
```
