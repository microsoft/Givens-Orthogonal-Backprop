# Givens Rotation Parametrized Orthogonal Backpropagation

Code for: https://arxiv.org/abs/2106.00003 Allows for gradient descent over rotational matrices. 



## Installation
PLease use python 3.8. You can replicate our development environment via conda, simply run:
> conda env create -f environment.yml

> conda activate rotMatDev

To setup the pytorch extension, go to the source folder ('cpp/' for serial execution on cpu, 'cuda/' for parallel execution on gpu) and run 
> python setup.py install

An example NN layer is included in the GivensRotations.py files in the source folders.



## How to file issues and get help

This project uses GitHub Issues to track bugs and feature requests. Please search the existing issues before filing new issues to avoid duplicates. For new issues, file your bug or feature request as a new Issue. We'll try to address impactful issues. Please feel free to reach out if you have any questions!
