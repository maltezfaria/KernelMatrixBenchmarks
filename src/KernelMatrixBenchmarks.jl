module KernelMatrixBenchmarks

using HMatrices
using StaticArrays
using LinearAlgebra
using LoopVectorization

include("kernels.jl")
include("benchs.jl")

end # module
