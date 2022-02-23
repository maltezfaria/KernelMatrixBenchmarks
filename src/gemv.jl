"""
    pygemv(y,H,x;threads=false)

Interface function to be called from `python` to compute `Hx` using `y` to write
the result. Both `x` and `y` should be `PyArray`s.
"""
function pygemv(y::PyArray,H::HMatrix,x::PyArray;threads=false)
    t = @elapsed mul!(y,H,x;threads)
    @debug t
    return nothing
end
