
"""

# Result
::Matrix{2, 2} each entry is for the combination for when Xi and Xj are 0 or 1. So the term mi[1, 2] is for Xi=0 and Xj=1, etc.
::Float the mutual information value
"""
function mi(Xi::VI, Xj::VI) where {VI <: AbstractVector{<:Integer}}
    @assert length(Xi) == length(Xj) "Xi and Xj must have the same length"
    n = length(Xi)

    # Joint probabilities
    pxy = zeros(2, 2)
    for l in 1:n
        pxy[Xi[l] + 1, Xj[l] + 1] += 1
    end
    pxy ./= n

    # Marginal
    px = vec(sum(pxy, dims = 2))
    py = vec(sum(pxy, dims = 1))

    terms = zeros(2, 2)
    for a in 1:2, b in 1:2

        if pxy[a, b] > 0 # Work around for log(0) = -Inf
            terms[a, b] = pxy[a, b] * log(pxy[a, b] / (px[a] * py[b]))
        end
    end
    return terms, sum(terms)
end

function imi(Xi::VI, Xj::VI) where {VI <: AbstractVector{<:Integer}}
    mi_res, _ = mi(Xi, Xj)
    return mi_res[1, 1] + mi_res[2, 2] - abs(mi_res[2, 1]) - abs(mi_res[1, 2])
end

function mimi(Xi::VI, Xj::VI) where {VI <: AbstractVector{<:Integer}}
    denom = entropy(Xi) + entropy(Xj)
    return denom == 0 ? 0.0 : 2 * imi(Xi, Xj) / denom
end

function entropy(X::VI) where {VI <: AbstractVector{<:Integer}}
    p1 = sum(X .== 1) / length(X)
    p0 = 1 - p1
    # Work around for log(0) = -Inf
    h = 0.0
    p0 > 0 && (h -= p0 * log(p0))
    p1 > 0 && (h -= p1 * log(p1))
    return h
end
