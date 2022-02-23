"""
    pyassemble(tag::String,X::PyArray,Y::PyArray)

Interface function to be called from `python` to assemble an `HMatrix` from point
clouds `X`  and `Y` given as a `n × d` and `m × d` python array, respectively,
where `d` is the ambient dimension and `m/n` are the number of target/source points.
"""
function pyassemble(tag::String,X::PyArray,Y::PyArray;nmax=200,eta=3,threads=false)
    # create a vector of Points
    @assert size(X,2) == size(Y,2) "ambient dimension of target points and source points must be equal"
    S = promote_type(eltype(X),eltype(Y))
    D = size(X,2)
    same_surface = X == Y
    @debug "same_surface: $same_surface"
    Xpts = [SVector{D,S}(c) for c in eachrow(X)]
    Ypts = same_surface ? Xpts : [SVector{D,S}(c) for c in eachrow(Y)]
    # construct cluster trees. Mutate Xpts and Ypts in place since these are
    # already local copies of X and Y
    spl  = CardinalitySplitter(;nmax)
    Xclt = ClusterTree(Xpts,spl;copy_elements=false)
    Yclt = same_surface ? Xclt : ClusterTree(Ypts,spl;copy_elements=false)
    if tag === "inverse-distance"
        K = InverseDistanceKernel(Xpts,Ypts)
    else
        error("unrecognized kernel type. Available options are $SUPPORTED_KERNELS")
    end
    # assemble the hmatrix
    adm = StrongAdmissibilityStd(;eta)
    return assemble_hmat(K,Xclt,Yclt;adm,global_index=false,threads)
end
