
abstract type AbstractTransmissionModel end
struct Exponential <: AbstractTransmissionModel end
struct PowerLaw <: AbstractTransmissionModel end
struct Rayleigh <: AbstractTransmissionModel end

@inline is_transmission_possible(::Exponential, Δ) = Δ > zero(Δ)
@inline is_transmission_possible(::Rayleigh, Δ) = Δ > zero(Δ)
@inline is_transmission_possible(::PowerLaw, Δ) = Δ > one(Δ)

@inline survival_weight(::Exponential, Δ) = Δ
@inline survival_weight(::PowerLaw, Δ) = Δ > one(Δ) ? log(Δ) : zero(Δ)
@inline survival_weight(::Rayleigh, Δ) = Δ^2 / 2

@inline hazard_weight(::Exponential, Δ) = one(Δ)
@inline hazard_weight(::PowerLaw, Δ) = inv(Δ)
@inline hazard_weight(::Rayleigh, Δ) = Δ

"""
    CascadeIndex

Pre-processed view of the cascades.

  - `order[c]`: infected nodes of cascade `c`, sorted by infection time.
  - `times[c]`: the matching (sorted) infection times.
  - `membership[i]`: `(c, position)` pairs for every cascade in which node `i` is infected.
  - `total_weight[j]`: `Σ_c w(T - t_j^c)` over cascades where `j` is infected, i.e. the
    survival term of `j` towards a node that stays uninfected in every cascade.
"""
struct CascadeIndex{T <: AbstractFloat}
    order::Vector{Vector{Int}}
    times::Vector{Vector{T}}
    membership::Vector{Vector{Tuple{Int, Int}}}
    total_weight::Vector{T}
end

"""
    NodeWorkspace{T}(num_nodes)

Scratch buffers reused across the per-node problems solved by a single task.
"""
struct NodeWorkspace{T <: AbstractFloat}
    weight::Vector{T}            # dense, indexed by global node id (lazily initialised)
    slot::Vector{Int}            # 0: untouched, -1: touched, k > 0: candidate number k
    touched::Vector{Int}
    candidates::Vector{Int}      # global ids of candidate parents
    group_offsets::Vector{Int}   # CSR-like offsets, one group per cascade (Ψ₃ term)
    hazard_index::Vector{Int}    # candidate number of each entry
    hazard_value::Vector{T}      # h(Δ) of each entry
    local_weight::Vector{T}      # w_j of each candidate
    rates::Vector{T}             # α_j of each candidate
    accumulator::Vector{T}
end

function NodeWorkspace{T}(num_nodes::Integer) where {T <: AbstractFloat}
    return NodeWorkspace{T}(zeros(T, num_nodes), zeros(Int, num_nodes), Int[], Int[], Int[],
        Int[], T[], T[], T[], T[])
end

function reset_workspace!(ws::NodeWorkspace)
    @inbounds for j in ws.touched
        ws.slot[j] = 0
    end
    empty!(ws.touched)
    empty!(ws.candidates)
    empty!(ws.hazard_index)
    empty!(ws.hazard_value)
    return nothing
end

function default_horizon(cascades::AbstractVector{<:AbstractVector{T}}) where {T <: AbstractFloat}
    horizon = zero(T)
    for cascade in cascades, t in cascade
        isfinite(t) && (horizon = max(horizon, t))
    end
    return horizon
end

function build_cascade_index(model::AbstractTransmissionModel,
        cascades::AbstractVector{<:AbstractVector{T}},
        horizon::T) where {T <: AbstractFloat}
    num_cascades = length(cascades)
    num_nodes = length(first(cascades))
    order = Vector{Vector{Int}}(undef, num_cascades)
    times = Vector{Vector{T}}(undef, num_cascades)
    membership = [Tuple{Int, Int}[] for _ in 1:num_nodes]
    total_weight = zeros(T, num_nodes)

    for c in 1:num_cascades
        cascade = cascades[c]
        infected = findall(<=(horizon), cascade)   # Inf (and NaN) are never infected
        sort!(infected; by = k -> @inbounds cascade[k])
        infection_times = cascade[infected]
        order[c] = infected
        times[c] = infection_times
        for (position, j) in enumerate(infected)
            push!(membership[j], (c, position))
            total_weight[j] += survival_weight(model, horizon - infection_times[position])
        end
    end
    return CascadeIndex{T}(order, times, membership, total_weight)
end

"""
Fill the workspace with the data of the convex problem of target node `i`.

Survival weights follow Eq. 10 without ever touching cascades in which `i` is absent:
starting from `total_weight[j]` (the Ψ₁ contribution of `j` in every cascade), each
cascade where `i` is infected replaces `j`'s Ψ₁ term by the Ψ₂ term `w(t_i - t_j)` when `j`
precedes `i`, or simply drops it otherwise (`i` is infected, so it is not a survivor).
Cascades where `i` has no admissible parent (e.g. roots) carry no Ψ₃ term and are skipped.
"""
function assemble_node_problem!(ws::NodeWorkspace{T}, model::AbstractTransmissionModel,
        index::CascadeIndex{T}, i::Integer, horizon::T) where {T <: AbstractFloat}
    (; weight, slot, touched, candidates, group_offsets, hazard_index, hazard_value) = ws
    empty!(group_offsets)
    push!(group_offsets, 1)

    for (c, position) in index.membership[i]
        order = index.order[c]
        times = index.times[c]
        t_i = times[position]
        num_before = searchsortedfirst(times, t_i) - 1   # strictly earlier infections

        @inbounds for k in eachindex(order)
            j = order[k]
            j == i && continue
            if slot[j] == 0
                slot[j] = -1
                weight[j] = index.total_weight[j]
                push!(touched, j)
            end
            t_j = times[k]
            weight[j] -= survival_weight(model, horizon - t_j)
            if k <= num_before
                Δ = t_i - t_j
                if is_transmission_possible(model, Δ)
                    weight[j] += survival_weight(model, Δ)
                    if slot[j] < 0
                        push!(candidates, j)
                        slot[j] = length(candidates)
                    end
                    push!(hazard_index, slot[j])
                    push!(hazard_value, hazard_weight(model, Δ))
                end
            end
        end
        # close the group of this cascade only if it received at least one parent
        length(hazard_index) + 1 > group_offsets[end] &&
            push!(group_offsets, length(hazard_index) + 1)
    end

    resize!(ws.local_weight, length(candidates))
    @inbounds for k in eachindex(candidates)
        # guard against round-off from the incremental weight bookkeeping
        ws.local_weight[k] = max(weight[candidates[k]], eps(T))
    end
    return nothing
end

"""
Solve the convex problem of one node with the monotone multiplicative update. Zero rates
(absent edges) emerge from the likelihood itself; no sparsity penalty is needed.
"""
function solve_node!(ws::NodeWorkspace{T}, tol::T, maxiter::Integer) where {T <: AbstractFloat}
    (; group_offsets, hazard_index, hazard_value, local_weight, rates, accumulator) = ws
    num_candidates = length(local_weight)
    num_groups = length(group_offsets) - 1
    resize!(rates, num_candidates)
    resize!(accumulator, num_candidates)
    fill!(rates, one(T))

    for _ in 1:maxiter
        fill!(accumulator, zero(T))
        @inbounds for g in 1:num_groups
            first_entry = group_offsets[g]
            last_entry = group_offsets[g + 1] - 1
            total = zero(T)
            for e in first_entry:last_entry
                total += hazard_value[e] * rates[hazard_index[e]]
            end
            iszero(total) && continue
            scale = inv(total)
            for e in first_entry:last_entry
                k = hazard_index[e]
                accumulator[k] += rates[k] * hazard_value[e] * scale
            end
        end
        max_change = zero(T)
        @inbounds for k in 1:num_candidates
            updated = accumulator[k] / local_weight[k]
            max_change = max(max_change, abs(updated - rates[k]))
            rates[k] = updated
        end
        max_change <= tol && break
    end
    return nothing
end

# function solve_node_lbfgs!(ws, tol, maxiter)
#     (; group_offsets, hazard_index, hazard_value, local_weight) = ws
#     k = length(local_weight)
#     function fg!(F, G, α)
#         G === nothing || copyto!(G, local_weight)
#         f = dot(local_weight, α)
#         for g in 1:(length(group_offsets) - 1)
#             r = group_offsets[g]:(group_offsets[g + 1] - 1)
#             s = sum(e -> hazard_value[e] * α[hazard_index[e]], r)
#             f -= log(s)
#             G === nothing || for e in r
#                 G[hazard_index[e]] -= hazard_value[e] / s
#             end
#         end
#         return f
#     end
#     res = optimize(Optim.only_fg!(fg!), fill(1e-8, k), fill(Inf, k), ones(k),
#                    Fminbox(LBFGS()), Optim.Options(g_tol = tol, outer_iterations = 20))
#     copyto!(resize!(ws.rates, k), Optim.minimizer(res))
# end

# Worker: pulls target nodes from a shared counter (dynamic load balancing) and returns
# the COO triplets of the inferred incoming edges.
function solve_nodes(model::AbstractTransmissionModel, index::CascadeIndex{T}, horizon::T,
        next_node::Threads.Atomic{Int}, tol::T, maxiter::Integer,
        edge_tol::T) where {T <: AbstractFloat}
    num_nodes = length(index.membership)
    ws = NodeWorkspace{T}(num_nodes)
    rows = Int[]
    cols = Int[]
    vals = T[]
    while true
        i = Threads.atomic_add!(next_node, 1)
        i > num_nodes && break
        assemble_node_problem!(ws, model, index, i, horizon)
        if !isempty(ws.candidates)
            solve_node!(ws, tol, maxiter)
            for k in eachindex(ws.candidates)
                α = ws.rates[k]
                if α > edge_tol
                    push!(rows, ws.candidates[k])
                    push!(cols, i)
                    push!(vals, α)
                end
            end
        end
        reset_workspace!(ws)
    end
    return rows, cols, vals
end


struct NetRate
    model::AbstractTransmissionModel
    tol::Real
    maxiter::Integer
    edge_tol::Real
    ntasks::Integer
    verbose::Bool
end
function NetRate(;
    model::AbstractTransmissionModel=Exponential(),
    tol::Real = 1e-8,
    maxiter::Integer = 5_000,
    edge_tol::Real = 1e-6,
    ntasks::Integer = Threads.nthreads(),
    verbose::Bool=false
)
    # if !(model in subtypes(AbstractTransmissionModel))
    #     error(
    #         "Likelihood model $model not avaliable. Please choose from $(repr(avliable_models_likelyhood))"*
    #         "If you are creating a custom model `f` make it inherit from the abstract type `AbstractTransmissionModel`"
    #     )
    # end
    NetRate(model, tol, maxiter, edge_tol, ntasks, verbose)
end


"""
    netrate(cascades, model; horizon, tol, maxiter, edge_tol, ntasks) -> SparseMatrixCSC

Infer the diffusion network and its transmission rates from `cascades`.

`cascades` is a vector of cascades; `cascades[c][k]` is the infection time of node `k` in
cascade `c`, with `Inf` for nodes that were not infected inside the observation window.
Every cascade must have the same length (the number of nodes) and a clock starting at 0.

# Keyword arguments

  - `horizon`: observation window `T`. Times larger than `horizon` count as uninfected.
    Defaults to the largest finite time (pass the real window when you know it: it
    controls how strongly never-infected nodes penalize edges, see Ψ₁).
  - `tol`: stop a node's solver when the largest absolute change of a rate is below `tol`.
  - `maxiter`: maximum number of iterations per node.
  - `edge_tol`: rates `≤ edge_tol` are reported as absent edges (the optimum is exactly
    zero there; the iterative solver only approaches it).
  - `ntasks`: number of parallel workers; the per-node problems are independent.

# Returns

A sparse matrix `A` with `A[j, i] = α_{j,i}`, the transmission rate of edge `j → i`.
"""
function solve(
    ie_problem::IEProblem,
    solver::NetRate
)
    T = typeof(ie_problem.d.diffusion_history |> first |> first)
    if T == BitVector
        error(
            "Expected diffusion history of type Vector{T} but got Vector{BitVector}."*
            "This indicates the cascade does not provide time of contamination"
        )
    end
    horizon = default_horizon(ie_problem.d.diffusion_history)
    isempty(ie_problem.d.diffusion_history) && throw(ArgumentError("at least one cascade is required"))
    num_nodes = length(first(ie_problem.d.diffusion_history))
    all(c -> length(c) == num_nodes, ie_problem.d.diffusion_history) ||
        throw(ArgumentError("all cascades must have the same length"))

    horizon_T = T(horizon)
    index = build_cascade_index(solver.model, ie_problem.d.diffusion_history, horizon_T)
    next_node = Threads.Atomic{Int}(1)
    tasks = map(1:solver.ntasks) do _
        Threads.@spawn solve_nodes(
            solver.model,
            index,
            horizon_T,
            next_node,
            T(solver.tol),
            solver.maxiter,
            T(solver.edge_tol)
        )
    end
    results = fetch.(tasks)

    rows = reduce(vcat, (r[1] for r in results))
    cols = reduce(vcat, (r[2] for r in results))
    vals = reduce(vcat, (r[3] for r in results))
    return sparse(rows, cols, vals, num_nodes, num_nodes)
end
