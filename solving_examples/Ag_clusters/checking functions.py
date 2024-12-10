# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:44:32 2024

@author: n.patsalidis
"""
import numpy as np
from numba import njit

# Setup
natoms = 250
atoms = np.zeros((natoms, 3))  # Array to store results
n_pairs = natoms * (natoms - 1) // 2
pairwise_forces = np.random.rand(n_pairs, 3)
pairs = np.array([(i, j) for i in range(natoms) for j in range(i + 1, natoms)])
i_indices, j_indices = pairs[:, 0], pairs[:, 1]

# NumPy add.at
def numpy_add(atoms, pairwise_forces, i_indices, j_indices):
    np.add.at(atoms, i_indices, pairwise_forces)
    np.add.at(atoms, j_indices, pairwise_forces)
    return atoms

# Serial Python loop
def python_add(atoms, pairwise_forces, i_indices, j_indices):
    for k in range(len(i_indices)):
        i, j = i_indices[k], j_indices[k]
        atoms[i] += pairwise_forces[k]
        atoms[j] += pairwise_forces[k]
    return atoms

# Numba-optimized loop
@njit
def numba_add(atoms, pairwise_forces, i_indices, j_indices):
    for k in range(len(i_indices)):
        i, j = i_indices[k], j_indices[k]
        atoms[i] += pairwise_forces[k]
        atoms[j] += pairwise_forces[k]
    return atoms
forces = np.zeros_like(atoms)
import time
data_points = 50
# Time NumPy
start = time.time()
for i in range(data_points):
    numpy_result = numpy_add(forces, pairwise_forces, i_indices, j_indices)
numpy_time = time.time() - start

# Time Python loop
forces = np.zeros_like(atoms)
start = time.time()
for i in range(10):
    python_result = python_add(forces, pairwise_forces, i_indices, j_indices)
python_time = time.time() - start
forces = np.zeros_like(atoms)
# Time Numba
start = time.time()
for i in range(data_points):
    numba_result = numba_add(forces, pairwise_forces, i_indices, j_indices)
numba_time = time.time() - start

# Compare results
print("NumPy time: {:4.3e} ms".format(  numpy_time*1000/data_points ) )
print("Python time:{:4.3e} ms".format(  python_time*1000/10 ))
print("Numba time:{:4.3e} ms".format(  numba_time*1000/data_points ))
print("Differences (NumPy vs Python):", np.allclose(numpy_result, python_result))
print("Differences (NumPy vs Numba):", np.allclose(numpy_result, numba_result))
