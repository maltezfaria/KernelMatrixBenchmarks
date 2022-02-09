# test the python interface
import julia
from julia import Main
from julia import KernelMatrixBenchmarks # assume that KernelMatrixBenchmarks is in the main julia environment
import numpy as np
import time

bench = KernelMatrixBenchmarks.Bench()
m     = 10000
X     = np.random.rand(m,3)

# assemble 
KernelMatrixBenchmarks.assemble("inverse_distance",bench,X)

# multiply
b     = np.random.rand(m)
out   = KernelMatrixBenchmarks.gemv(bench,b)