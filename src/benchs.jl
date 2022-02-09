"""
    mutable struct Bench

A structure encapsulating a given benchmark from
[kernel-matrix-benchmarks](https://github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks.git) repo.
"""
mutable struct Bench
    hmat
end
Bench() = Bench(nothing)

function gemv(b::Bench,X::Vector)
    H      = b.hmat
    T      = eltype(H)
    out    = zeros(T,size(H,1))
    mul!(out,b.hmat,X)
    return out
end

function assemble(bench_type,b::Bench,X::Matrix,Y::Matrix=X)
    if bench_type == "inverse_distance"
        _assemble_inverse_distance(b,X,Y)
    else
        error("unrecognized bench specification")
    end
end

function _assemble_inverse_distance(b,X::Matrix,Y::Matrix)
    @assert size(X,2) == size(Y,2)
    @assert eltype(X) == eltype(Y)
    S = eltype(X)
    D = size(X,2)
    # create a vector of Points
    Xpts = [SVector{D,S}(c) for c in eachrow(X)]
    Ypts = [SVector{D,S}(c) for c in eachrow(Y)]
    # construct cluster trees. Mutate Xpts and Ypts in place since these are
    # already local copies of X and Y
    nmax = 200
    spl  = CardinalitySplitter(;nmax)
    Xclt = ClusterTree(Xpts,spl;copy_elements=false)
    Yclt = ClusterTree(Ypts,spl;copy_elements=false)
    # construct kernel on permuted indices
    K    = InverseDistanceKernel(Xpts,Ypts)
    # assemble the hmatrix
    eta = 3
    adm    = StrongAdmissibilityStd(;eta)
    b.hmat = assemble_hmat(K,Xclt,Yclt;adm,global_index=false,threads=false)
    return b
end
