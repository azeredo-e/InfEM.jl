# log-likelihood contribution of a single node i (Eqs. 7-9, summed over all processes)
# Perturbing only a subset of nodes Clause hallucination, later test if it does anything good
function node_loglik(i, P, incoming, diffusion_res)
    β = size(diffusion_res, 2)
    ll = 0.0
    for l in 1:β
        if diffusion_res[i, l] == 1
            s_inf = 0.0
            s_not = 0.0
            for (j, k) in incoming[i]
                if diffusion_res[j, l] == 1
                    s_inf += log1p(-P[k])
                else
                    s_not += log1p(-P[k])
                end
            end
            s_inf != 0.0 && (ll += log(-expm1(s_inf)))   # Eq. (7)
            ll += s_not                                  # Eq. (8)
        else
            for (j, k) in incoming[i]
                diffusion_res[j, l] == 1 && (ll += log1p(-P[k]))   # Eq. (9)
            end
        end
    end
    return ll
end
function sa_optimize_sparse(P_init, directed, incoming, diffusion_res;
        t0 = 100.0, tf = 1.0, L = 200, K = 8,
        σ = 0.05, m = 3)
    n = size(diffusion_res, 1)
    x_c = copy(P_init)

    nll = [node_loglik(i, x_c, incoming, diffusion_res) for i in 1:n]  # per-node cache
    O_c = -sum(nll)                      # objective = negative log-likelihood
    x_b = copy(x_c)
    O_b = O_c
    t = t0
    k_temp = 1

    while t > tf
        for _ in 1:L
            ks = rand(1:length(x_c), m)         # edges to perturb
            old = x_c[ks]                        # saved for undo
            for k in ks
                x_c[k] = clamp(x_c[k] + σ * randn(), 1e-9, 1 - 1e-9)
            end

            affected = unique(directed[k][2] for k in ks)   # target nodes whose terms changed
            new_nll = [node_loglik(i, x_c, incoming, diffusion_res) for i in affected]
            Δ = sum(nll[i] for i in affected) - sum(new_nll)   # O_new - O_current

            if Δ < 0 || rand() < exp(-Δ / t)
                for (idx, i) in enumerate(affected)
                    nll[i] = new_nll[idx]
                end
                O_c += Δ
                if O_c < O_b
                    x_b .= x_c
                    O_b = O_c
                end
            else
                x_c[ks] .= old                   # reject: undo the perturbation
            end
        end
        t = k_temp < K ? t0 / log(k_temp + 1) : t0 * 0.99^k_temp
        k_temp += 1
    end
    return x_b, -O_b
end

@doc LIPN_DOC
struct LIPN{
    F <: AbstractFloat,
    N <: Number,
    I <: Integer
} <: AbstractIEPSolver
    e_threshold::F
    em_tol::F
    em_maxiter::I
    sa_temp_ini::N
    sa_temp_final::N
    sa_epochs::I
    sa_temp_decay::N
    verbose::Bool
end
function LIPN(;
        e_threshold::F = 0.5, # For LIPN
        em_tol::F = 1e-6,
        em_maxiter::I = 100,
        sa_temp_ini::N = 100, # For SA
        sa_temp_final::N = 1,
        sa_epochs::I = 200,
        sa_temp_decay::N = 8,
        verbose::Bool = false
) where {
        F <: AbstractFloat,
        N <: Number,
        I <: Integer
}
    LIPN(
        e_threshold,
        em_tol,
        em_maxiter,
        sa_temp_ini,
        sa_temp_final,
        sa_epochs,
        sa_temp_decay,
        verbose
    )
end

function _log_likelihood(
        P::Vector{Float64},
        incoming::Vector{Vector{Tuple{Int64, Int64}}},
        diffusion_res::BitMatrix
)
    n, D_d = size(diffusion_res)
    ll = 0.0
    for l in 1:D_d
        for i in 1:n
            if diffusion_res[i, l] == 1
                s_inf = 0.0
                s_not = 0.0
                for (j, k) in incoming[i]
                    if diffusion_res[j, l] == 1
                        s_inf += log1p(-P[k]) # N^++, productory part
                    else
                        s_not += log1p(-P[k]) # N^-+
                    end
                end
                if s_inf != 0.0 # Fix so that I calculate *1 -* the productory
                    ll += log1p(-exp(s_inf))
                end
                ll += s_not
            else
                for (j, k) in incoming[i]
                    diffusion_res[j, l] == 1 && (ll += log1p(-P[k])) # N^+-
                end
            end
        end
    end
    return ll
end

function _combined_temperature_sa(
        P_init::Vector{Float64},
        incoming::Vector{Vector{Tuple{Int64, Int64}}},
        diffusion_res::BitMatrix;
        params::LIPN # Solver defined params
)
    f(P) = -_log_likelihood(P, incoming, diffusion_res)   # minimize negative log-lik

    x_c = copy(P_init)
    O_c = f(x_c)
    x_b = copy(x_c)
    O_b = O_c
    t = params.sa_temp_ini
    k = 1

    while t > params.sa_temp_final
        for _ in 1:params.sa_epochs
            e = rand(Normal(), length(x_c))
            x_n = clamp.(x_c + e, 1e-9, 1 - 1e-9) # Add some noise to data to see how the likelihood changes
            O_n = f(x_n)
            if O_n < O_c || rand() < exp((O_c - O_n) / t)
                x_c, O_c = x_n, O_n
                if O_c < O_b
                    x_b, O_b = copy(x_c), O_c
                end
            end
        end
        t = k < params.sa_temp_decay ? params.sa_temp_ini / log(k + 1) :
            params.sa_temp_ini * 0.99^k
        k += 1
    end
    return x_b, -O_b
end

function _e_step(
        P::Vector{Float64},
        incoming::Vector{Vector{Tuple{Int64, Int64}}},
        diffusion_res::BitMatrix
)
    n, D_d = size(diffusion_res)
    # l -> i -> [(j, k, γ), ...] :: for each neighbor of i for each diffusion
    γ = [Dict{Int, Vector{Tuple{Int, Int, Float64}}}() for _ in 1:D_d]
    for l in 1:D_d, i in 1:n

        diffusion_res[i, l] == 1 || continue
        prod_q = 1.0
        for (j, k) in incoming[i] # Its the same principle as when calculating the log likelihood
            diffusion_res[j, l] == 1 && (prod_q *= 1 - P[k])
        end
        denom = 1 - prod_q
        denom <= 0 && continue
        vec = Tuple{Int, Int, Float64}[]
        for (j, k) in incoming[i]
            diffusion_res[j, l] == 1 && push!(vec, (j, k, P[k] / denom))
        end
        γ[l][i] = vec
    end
    return γ
end

function _m_step(
        γ::Vector{Dict{Int, Vector{Tuple{Int, Int, Float64}}}},
        P::Vector{Float64},
        D_d::Integer
)
    sum_γ = zeros(length(P))
    cnt = zeros(Int, length(P))

    for l in 1:D_d
        for (i, entries) in γ[l]
            for (j, k, γ) in entries
                sum_γ[k] += γ # Numerator
                cnt[k] += 1 # Diffusion where i activated
            end
        end
    end

    P_new = similar(P)
    for k in eachindex(P)
        D_minus = D_d - cnt[k] # Diffusion where i wasnt activated
        P_new[k] = sum_γ[k] / (sum_γ[k] + D_minus)
    end
    return P_new
end

function solve(ie_problem::IEProblem, solver::LIPN)::IESolution
    t0 = time()
    diffusion_res = reduce(hcat, ie_problem.d.diffusion_history) # Each column is the result of some diffusion process
    n, D_d = size(diffusion_res)

    # Pruning process, remove lesse likely edges based on the MIMI measure
    pairs = [(i, j) for i in 1:n for j in (i + 1):n] # upper-triangle pairs only, computed once
    vals = Vector{Float64}(undef, length(pairs))

    @views for (k, (i, j)) in enumerate(pairs)
        vals[k] = mimi(diffusion_res[i, :], diffusion_res[j, :])
    end
    τ = _prune_threshold(vals) # utils.jl
    candidates = pairs[vals .>= τ] # Vector of tuples (possible edges (v, u)) that survived the pruning

    # Initial probabilities, LIPN assumes all connections are bidirectional
    directed = Vector{Tuple{Int, Int}}(undef, 2 * length(candidates))
    for (k, (i, j)) in enumerate(candidates)
        directed[2k - 1] = (i, j)
        directed[2k] = (j, i)
    end
    P0 = rand(length(directed))

    # Vector of Vector of Tuples: for each node a Vector of tuples, each tuple is one neighbor and their
    # associated initial prob. for that edge
    incoming = [Tuple{Int, Int}[] for _ in 1:n]
    for (k, (src, dst)) in enumerate(directed)
        push!(incoming[dst], (src, k))
    end

    # Initial likelihood calculation
    L0 = _log_likelihood(P0, incoming, diffusion_res)

    # EM otimization
    P_edges = copy(P0) # Edge probabilities
    epoch = 0
    while (solver.em_maxiter < epoch)
        γ_hat = _e_step(P_edges, incoming, diffusion_res)
        P_new = _m_step(γ_hat, P_edges, D_d)
        # Work around so that we dont get log(0) when calculating the log-likelihood for the SA algorithm
        P_new .= clamp.(P_new, 1e-9, 1 - 1e-9)
        maximum(abs.(P_new .- P_edges)) < solver.em_tol && break
        P_edges = P_new
        epoch += 1
    end

    # SA procedure to approximate to global minimum
    L_em = _log_likelihood(P_edges, incoming, diffusion_res)
    if L_em > L0
        P_edges, _ = _combined_temperature_sa(P_edges, incoming, diffusion_res; params = solver)
    else
        P_edges, _ = _combined_temperature_sa(P0, incoming, diffusion_res; params = solver)
    end

    # Final result
    keep = P_edges .>= solver.e_threshold
    E = directed[keep]
    P_edges = P_edges[keep]

    return IESolution(
        ie_problem.g,
        ie_problem.d,
        (E, P_edges),
        NamedTuple{(:start, :end, :elapsed)}((t0, time(), time() - t0))
    )
end
