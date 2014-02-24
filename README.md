tucker3d Version 0.1

tucker3d
========

"tucker3d" is Python implementation of the Tucker format in 3 dimensional case.
This implemetation includes 
- Basic linear algebra operations
- New version of the cross approximation method (it allows to construct tucker representation 
of a tensor by using only few of its elements)

Installation
============

Clone this repository, then go to core/maxvol and run
```
python setup.py build_ext —inplace
```
