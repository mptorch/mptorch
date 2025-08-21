.. mptorch documentation master file, created by
   sphinx-quickstart on Fri Sep 20 16:17:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the MPTorch documentation!
==============================================================


**MPTorch** is a PyTorch-based framework that is designed to simulate the use of custom/mixed precision
arithmetic in PyTorch, in particular for DNN training workflows. It offers quantization support for various
number formats (fixed-point, floating-point and block floating-point based representations) and 
reimplements the underlying computations of commonly used DNN layers (in CNN and Transformer-based
models) using user-specified formats for each operation (e.g. addition, multiplication). All operations 
are internally done using IEEE-754 32-bit floating-point arithmetic, with the results rounded to the 
specified format.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   tutorial
   examples
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
