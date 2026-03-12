using NLPModels
using MadNLP
using LinearAlgebra

include("step_model.jl")
using .SubP2StepModel

struct SubP2NoHessianModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    step::StepData
    counters::NLPModels.Counters
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    grad_weights::Vector{T}
    grad_target::Vector{T}
end

function SubP2NoHessianModel(step::StepData; T = Float64)
    n = length(step.x_init)
    meq = length(step.init_eq_ref)
    mineq = length(step.init_ineq_ref)
    m = meq + mineq
    nnzj = m * n
    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    k = 1
    for i in 1:m, j in 1:n
        rows[k] = i
        cols[k] = j
        k += 1
    end
    lcon = vcat(zeros(T, meq), fill(T(-Inf), mineq))
    ucon = zeros(T, m)
    meta = NLPModels.NLPModelMeta(
        n,
        ncon = m,
        nnzj = nnzj,
        nnzh = 0,
        x0 = T.(step.x_init),
        y0 = zeros(T, m),
        lvar = fill(T(-Inf), n),
        uvar = fill(T(Inf), n),
        lcon = lcon,
        ucon = ucon,
        minimize = true,
        sparse_jacobian = true,
        sparse_hessian = true,
    )
    weights, target = objective_quadratic_model(step, T)
    return SubP2NoHessianModel(meta, step, NLPModels.Counters(), rows, cols, weights, target)
end

@inline function fd_step(xj)
    return cbrt(eps(Float64)) * max(1.0, abs(xj))
end

function objective_quadratic_model(step::StepData, ::Type{T}) where T
    nxl, nul, nxi, nui, nq, _ = step.dims
    active_u = T(1.0 - Float64(step.params["is_terminal"]))
    rho_lx = T(Float64(step.params["rho_lx"]))
    rho_lu = T(Float64(step.params["rho_lu"]))
    rho_ix = T(Float64(step.params["rho_ix"]))
    rho_iu = T(Float64(step.params["rho_iu"]))

    x_target = zeros(T, length(step.x_init))
    weights = zeros(T, length(step.x_init))

    xl_ideal = T.(as_vec(step.params["xl_ideal"]))
    ul_ideal = T.(as_vec(step.params["ul_ideal"]))
    y_xl = T.(as_vec(step.params["y_xl"]))
    y_ul = T.(as_vec(step.params["y_ul"]))
    xc_ideal = T.(rowmajor_vec(as_mat(step.params["xc_ideal"])))
    uc_ideal = T.(rowmajor_vec(as_mat(step.params["uc_ideal"])))
    y_xc = T.(rowmajor_vec(as_mat(step.params["y_xc"])))
    y_uc = T.(rowmajor_vec(as_mat(step.params["y_uc"])))

    x_target[1:nxl] .= xl_ideal .- y_xl ./ (rho_lx + T(1e-6))
    weights[1:nxl] .= rho_lx

    ul_lo = nxl + 1
    ul_hi = nxl + nul
    x_target[ul_lo:ul_hi] .= ul_ideal .- y_ul ./ (rho_lu + T(1e-6))
    weights[ul_lo:ul_hi] .= active_u * rho_lu

    base = nxl + nul
    xc_len = nq * nxi
    xc_lo = base + 1
    xc_hi = base + xc_len
    x_target[xc_lo:xc_hi] .= xc_ideal .- y_xc ./ (rho_ix + T(1e-6))
    weights[xc_lo:xc_hi] .= rho_ix

    uc_lo = xc_hi + 1
    uc_hi = uc_lo + nq * nui - 1
    x_target[uc_lo:uc_hi] .= uc_ideal .- y_uc ./ (rho_iu + T(1e-6))
    weights[uc_lo:uc_hi] .= active_u * rho_iu

    return weights, x_target
end

function cons_vec(step::StepData, x::AbstractVector)
    eq = equality_residual(x, step.params, step.dims)
    ineq = inequality_residual(x, step.params, step.dims)
    return vcat(Float64.(eq), Float64.(ineq))
end

function NLPModels.obj(nlp::SubP2NoHessianModel, x::AbstractVector)
    return objective(x, nlp.step.params, nlp.step.dims)
end

function NLPModels.grad!(nlp::SubP2NoHessianModel, x::AbstractVector, g::AbstractVector)
    @inbounds for j in eachindex(g)
        g[j] = nlp.grad_weights[j] * (x[j] - nlp.grad_target[j])
    end
    return g
end

function NLPModels.cons!(nlp::SubP2NoHessianModel, x::AbstractVector, c::AbstractVector)
    cv = cons_vec(nlp.step, x)
    copyto!(c, cv)
    return c
end

function NLPModels.jac_structure!(nlp::SubP2NoHessianModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, T.(nlp.jac_rows))
    copyto!(J, T.(nlp.jac_cols))
    return I, J
end

function NLPModels.jac_coord!(nlp::SubP2NoHessianModel, x::AbstractVector, J::AbstractVector)
    xw = collect(Float64, x)
    c0 = cons_vec(nlp.step, xw)
    m = length(c0)
    n = length(xw)
    k = 1
    for j in 1:n
        h = fd_step(xw[j])
        xp = copy(xw); xp[j] += h
        xm = copy(xw); xm[j] -= h
        cp = cons_vec(nlp.step, xp)
        cm = cons_vec(nlp.step, xm)
        col = (cp .- cm) ./ (2h)
        for i in 1:m
            J[k] = col[i]
            k += 1
        end
    end
    return J
end

function NLPModels.jprod!(nlp::SubP2NoHessianModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    xw = collect(Float64, x)
    vv = collect(Float64, v)
    α = cbrt(eps(Float64)) / max(1.0, norm(vv))
    xp = xw .+ α .* vv
    xm = xw .- α .* vv
    cp = cons_vec(nlp.step, xp)
    cm = cons_vec(nlp.step, xm)
    copyto!(jv, (cp .- cm) ./ (2α))
    return jv
end

function NLPModels.jtprod!(nlp::SubP2NoHessianModel, x::AbstractVector, v::AbstractVector, jtv::AbstractVector)
    xw = collect(Float64, x)
    vv = collect(Float64, v)
    ϕ(z) = dot(vv, cons_vec(nlp.step, z))
    for j in eachindex(xw)
        h = fd_step(xw[j])
        xp = copy(xw); xp[j] += h
        xm = copy(xw); xm[j] -= h
        jtv[j] = (ϕ(xp) - ϕ(xm)) / (2h)
    end
    return jtv
end

function solve_nohessian_step(
    path::String;
    print_level = MadNLP.ERROR,
    callback = MadNLP.SparseCallback,
    kkt_system = MadNLP.SparseKKTSystem,
    hessian_approximation = MadNLP.CompactLBFGS,
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
)
    step = load_step_data(path)
    nlp = SubP2NoHessianModel(step)
    t0 = time_ns()
    result = madnlp(
        nlp;
        print_level = print_level,
        callback = callback,
        kkt_system = kkt_system,
        hessian_approximation = hessian_approximation,
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    wall_ms = (time_ns() - t0) / 1e6
    x_sol = vec(Float64.(result.solution))
    eq = Float64.(equality_residual(x_sol, step.params, step.dims))
    ineq = Float64.(inequality_residual(x_sol, step.params, step.dims))
    return Dict(
        "status" => string(result.status),
        "iter" => result.iter,
        "wall_ms" => wall_ms,
        "objective" => objective(x_sol, step.params, step.dims),
        "eq_inf" => maximum(abs.(eq)),
        "ineq_vio" => maximum(max.(ineq, 0.0)),
        "orig_rmse" => sqrt(sum(abs2, x_sol .- step.x_orig) / length(x_sol)),
        "orig_max_abs" => maximum(abs.(x_sol .- step.x_orig)),
    )
end
