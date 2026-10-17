@doc TENDS_DOC
struct TENDS
    verbose::Bool
end
function TENDS(; verbose::Bool = false)
    TENDS(verbose)
end

function _local_score(i, F, S)
    D_d = size(S, 2)
    m = length(F)
    N = zeros(Int, 2^m, 2)
    for l in 1:D_d
        j = 1
        for (b, p) in enumerate(F)
            j += Int(S[p, l]) << (b - 1) # encode parent states as an index
        end
        N[j, Int(S[i, l]) + 1] += 1
    end
    ll = 0.0
    pen = 0.0
    φ = 0
    for j in 1:(2 ^ m)
        nj = N[j, 1] + N[j, 2]
        nj == 0 && (φ += 1)
        for k in 1:2
            N[j, k] > 0 && (ll += N[j, k] * log2(N[j, k] / nj))
        end
        pen += log2(nj + 1)
    end
    return ll - pen / 2, φ
end

function _delta_bound(N1, N2, D_d)
    t(N) = N == 0 ? 0.0 : 2N * log2(D_d / N)
    return t(N1) + t(N2) + log2(D_d + 1)
end

function _greedy_parents(i, P_i, S)
    D_d = size(S, 2)
    N2 = sum(S[i, :]) # Infected count
    N1 = D_d - N2 # Not infected count
    δ = _delta_bound(N1, N2, D_d)

    F = Int[]
    best, _ = _local_score(i, F, S)
    cand = copy(P_i)
    while !isempty(cand)
        bs, bj = -Inf, 0
        for c in cand
            F2 = vcat(F, c)
            s, φ = _local_score(i, F2, S)
            if length(F2) <= log2(φ + δ) && s > bs
                bs, bj = s, c
            end
        end
        bs > best || break # no improvement: stop
        push!(F, bj)
        filter!(!=(bj), cand)
        best = bs
    end
    return F
end

function solve(ie_problem::IEProblem, solver::TENDS)::IESolution
    t0 = time()
    diffusion_res = reduce(hcat, ie_problem.d.diffusion_history) # Each column is the result of some diffusion process
    n, _ = size(diffusion_res)
    E = Tuple{Int, Int}[] # Possible directed edges (parent, child)

    # Pruning process, remove less likely edges based on the IMI measure
    pairs = [(i, j) for i in 1:n for j in (i + 1):n] # upper-triangle pairs only, computed once
    vals = Vector{Float64}(undef, length(pairs))
    @views for (k, (i, j)) in enumerate(pairs)
        vals[k] = imi(diffusion_res[i, :], diffusion_res[j, :])
    end
    nonneg_vals = filter(>=(0), vals)

    if isempty(nonneg_vals) || all(iszero, nonneg_vals)
        @warn "No informative IMI values: no candidate parents"
    else
        τ = _prune_threshold(nonneg_vals)

        # Candidate parent set P_i of each node (IMI is symmetric)
        parents_cand = [Int[] for _ in 1:n]
        for (k, (i, j)) in enumerate(pairs)
            if vals[k] > τ
                push!(parents_cand[i], j)
                push!(parents_cand[j], i)
            end
        end

        # Greedy search for the most probable parents of each node
        for i in 1:n
            F = _greedy_parents(i, parents_cand[i], diffusion_res) # passar uma view só do que i que queremos do diffusion_res
            append!(E, [(j, i) for j in F]) # directed edge j → i
        end
    end

    return IESolution(
        ie_problem.g,
        ie_problem.d,
        E, # Vector{Tuple{Int,Int}}, each tuple is (parent, child), i.e. j → i
        NamedTuple{(:start, :end, :elapsed)}((t0, time(), time() - t0))
    )
end
