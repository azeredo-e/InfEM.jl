function _prune_threshold(mimi_vals::Vector{Float64}; maxiter = 100)
    # Used in LIPN and TENDS
    c = maximum(mimi_vals) / 2
    low = falses(length(mimi_vals))
    for _ in 1:maxiter
        # nearer to 0 than to c
        low_new = abs.(mimi_vals) .<= abs.(mimi_vals .- c)
        low_new == low && break
        low = low_new
        high = mimi_vals[.!low]
        isempty(high) && break
        # update only the free centroid
        c = sum(high) / length(high)
    end
    return maximum(mimi_vals[low]) # τ
end
