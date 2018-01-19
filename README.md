# AutoWhiteBalance
efficient and robust white balance algorithm

## Introduction
Try to implement google's white balance paper:

> Barron, Jonathan T. "Convolutional color constancy." In Proceedings of the IEEE International Conference on Computer Vision, pp. 379-387. 2015.

> Barron, Jonathan T., and Yun-Ta Tsai. "Fast Fourier Color Constancy." arXiv preprint arXiv:1611.07596 (2016).

## Implmentation

1. Use tensorflow as a tool for optimization. (I tried, but it seems that tensorflow optimizer can not solve this optimization problem well.)

2. Implement an optimization solver based on google's ffcc (the open source code of the second paper). The original code has many redundant code, I tried to re-implement a much more clean and easy-use version here.

3. A fast CUDA based white balance algorithm 

## How to use
1. download the training data and pre-trained model and extract to the root dir:
    link: https://pan.baidu.com/s/1jKeQWKm passwd: h2v5

2. training code is in ./matlab_training

3. C++/CUDA code used to apply auto white balance on input image is in ./Cpp

4. pre-trained model is in ./data/model (you should download it from baiduyun)
