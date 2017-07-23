### Intro

First release of  UberNet in "test" mode.
The demo performs all of the UberNet tasks using a VGG16-architecture network.

This repository builds on the py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn) release of faster-rcnn, which in turn builds on the caffe library. The py-faster-rcnn instructions directly allow one to compile the present code (copied below for convenience).
The main changes to the original py-faster-rcnn release have been annotated with
a <iasonas> prefix and a </iasonas> suffix.

Forthcoming:
- code that allows to get normalized cut eigenvectors from image boundaries.
- training code.
- ResNet-based models.

### License
UberNet is released under the GPL License (refer to the UberNet-LICENSE file for details).
Faster-RCNN is released under the MIT License (refer to the Faster-RCNN-LICENSE file for details).

### Citing UberNet

If you find UberNet useful in your research, please consider citing:
@inproceedings{ubernet,
    Author = {Iasonas Kokkinos},
    Title = {UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory},
    Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    Year = {2017}
}

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!
  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download my [Makefile.config](https://www.dropbox.com/s/zdsqped1h8yzbwg/Makefile.config?dl=0) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware
You will need a GPU to run the code (does not work in CPU-only mode). The present code is memory-efficient in the forward pass, but you may modify the .prototxt file to further reduce memory usage (please consult the paper, ubernet/test.prototxt and net.cpp to understand how the "deletetop", "deletebottom" variables are used)

### Installation (sufficient for the demo)

1. Clone the UberNet repository
  ```Shell
  git clone https://github.com/jkokkin/UberNet.git
  ```

2. We'll call the directory that you cloned UberNet into `UBERNET_ROOT`

3. Build the Cython modules
    ```Shell
    cd $UBERNET_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $UBERNET_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download a trained UberNet model from here:
   https://www.dropbox.com/s/fbeg10aoicn4wc4/model.caffemodel?dl=0
   and place it under $UBERNET_ROOT/model

### Demo
To run the demo
```Shell
cd $UBERNET_ROOT
./demo/demo_ubernet.py
```
