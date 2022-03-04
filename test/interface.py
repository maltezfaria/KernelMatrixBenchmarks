# This file tests the pythong interface for the assembling and gemv routines

import juliacall
from juliacall import Main as jl
import numpy as np
import time

# load packages
jl.seval("using Pkg"); 
jl.Pkg.activate("."); 
jl.Pkg.status()
jl.seval("using KernelMatrixBenchmarks")
jl.seval("using HMatrices")
jl.seval(""" ENV["JULIA_DEBUG"] = "KernelMatrixBenchmarks" """)

jl.HMatrices.disable_getindex()
jl.BLAS.set_num_threads(1)

m     = 10000
X     = np.random.rand(m,3)
Xf    = np.asfortranarray(X)

start = time.time()
hmat = jl.KernelMatrixBenchmarks.pyassemble("inverse-distance",Xf,Xf,threads=True,nmax=200)
end = time.time()
print("Assemble time:", end - start)

x = np.random.rand(m)
y = np.zeros(m)

start = time.time()
jl.KernelMatrixBenchmarks.pygemv(y,hmat,x)
end = time.time()
print("gemv time: ", end - start)
