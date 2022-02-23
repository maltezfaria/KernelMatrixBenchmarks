module KernelMatrixBenchmarks

using HMatrices
using StaticArrays
using LinearAlgebra
using LoopVectorization
using PythonCall

include("kernels.jl")
include("assemble.jl")
include("gemv.jl")

end # module
