tucker3d (Ver. 0.1)
===================

tucker3d is Python implementation of the Tucker format in the three-dimensional case.
This implemetation includes 
- Basic linear algebra operations
- New version of the cross approximation method (it allows to construct tucker representation 
of a tensor by using only few of its elements)
- Element-wise functions (multifun in the cross module)
- 3D convolutions (cross-conv algorithm, see http://arxiv.org/pdf/1402.5649.pdf for details)

Tucker format
=============

Tucker format is a low-parametric representation of multidimensional arrays (tensors).
This representaion is based on the idea of separation of variables (so-called tensor format).

If a Tucker representation of some tensors is given and it has low number of parameters, then basic linear algebra operations are fast to compute.


Installation
============

First, to clone this repository run
```
git clone git://github.com/rakhuba/tucker3d
```
Then go to 'maxvol' directory:
```
cd tucker3d/core/maxvol
```
and run
```
python setup.py build_ext --inplace
```
