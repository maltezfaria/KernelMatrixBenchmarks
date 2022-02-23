const SUPPORTED_KERNELS = ["inverse-distance"]

"""
    struct InverseDistanceKernel{T,Td} <: AbstractKernelMatrix{T}

Lazy representation of a matrix `A` giving the inverse of the distance between
point clouds `X` and `Y`.
"""
struct InverseDistanceKernel{T,Td} <: AbstractKernelMatrix{T}
    X::Matrix{Td} # n × 3
    Y::Matrix{Td} # m × 3
    function InverseDistanceKernel{T}(X::Matrix{Td}, Y::Matrix{Td}) where {T,Td}
        @assert size(X, 2) == size(Y, 2) == 3
        new{T,Td}(X, Y)
    end
end
InverseDistanceKernel(X::Matrix{T},Y::Matrix{T}) where {T} = InverseDistanceKernel{T}(X,Y)

# constructor from vector of points
function InverseDistanceKernel(_X::Vector{T},_Y::Vector{T}) where {T<:SVector}
    @assert length(T) === 3
    S = eltype(T)
    X = reshape(reinterpret(S,_X), 3,:) |> transpose |> collect
    Y = reshape(reinterpret(S,_Y), 3,:) |> transpose |> collect
    InverseDistanceKernel(X,Y)
end

Base.size(K::InverseDistanceKernel) = size(K.X, 1), size(K.Y, 1)

function Base.getindex(K::InverseDistanceKernel{T}, i::Int, j::Int)::T where {T}
    d2 = (K.X[i,1] - K.Y[j,1])^2 + (K.X[i,2] - K.Y[j,2])^2 + (K.X[i,3] - K.Y[j,3])^2
    return (d2!=0)*inv(sqrt(d2))
end
function Base.getindex(K::InverseDistanceKernel, I::UnitRange, J::UnitRange)
    T = eltype(K)
    m  = length(I)
    n  = length(J)
    Xv = view(K.X, I, :)
    Yv = view(K.Y, J, :)
    out = Matrix{T}(undef, m, n)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (Xv[i,1] - Yv[j,1])^2
            d2 += (Xv[i,2] - Yv[j,2])^2
            d2 += (Xv[i,3] - Yv[j,3])^2
            out[i,j] = (d2!=0)*inv(sqrt(d2))
        end
    end
    return out
end
function Base.getindex(K::InverseDistanceKernel, I::UnitRange, j::Int)
    return vec(K[I,j:j])
end
function Base.getindex(adjK::Adjoint{<:Any,InverseDistanceKernel}, I::UnitRange, j::Int)
    K = parent(adjK)
    vec(K[j:j,I])
end
