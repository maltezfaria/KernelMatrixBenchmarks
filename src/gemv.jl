"""
    pygemv(y::PyArray,H,x::PyArray;threads=false)

Interface function to be called from `python` to compute `Hx` using `y` to write
the result. Both `x` and `y` should be `PyArray`s.
"""
function pygemv(y::AbstractVector,H,x::AbstractVector;threads=false)
    t = @elapsed mul!(y,H,x;threads)
    @debug "gemv time: $t seconds"
    return y
end
