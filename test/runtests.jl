using Test
using KernelMatrixBenchmarks
using StaticArrays
using HMatrices

m = 10000
d = 3
X = Y = rand(m,d)

bench = KernelMatrixBenchmarks.Bench()
KernelMatrixBenchmarks.assemble("inverse_distance",bench,X,Y)

a = rand(m)
tmp = KernelMatrixBenchmarks.gemv(bench,a)
