if !isdefined(Main, :__SUBP2_JULIA_ENV_ACTIVE__)
    import Pkg
    Pkg.activate(@__DIR__)
    const __SUBP2_JULIA_ENV_ACTIVE__ = true
end

using JSON3
using JuMP
using MadNLP
using NLPModelsJuMP

include("step_model.jl")
using .SubP2StepModel: as_vec,
    as_mat,
    equality_residual,
    inequality_residual,
    load_step_data,
    objective,
    pair_term_indexed,
    gio_terms_indexed,
    thrust_terms_indexed

function rmse(a::AbstractVector, b::AbstractVector)
    return sqrt(sum(abs2, a .- b) / length(a))
end

function summarize_solution(step::SubP2StepModel.StepData, x_sol::Vector{Float64})
    eq_vec = Float64.(equality_residual(x_sol, step.params, step.dims))
    ineq_vec = Float64.(inequality_residual(x_sol, step.params, step.dims))
    return (
        objective = SubP2StepModel.objective(x_sol, step.params, step.dims),
        eq_inf = maximum(abs.(eq_vec)),
        ineq_vio = maximum(max.(ineq_vec, 0.0)),
        orig_rmse = rmse(x_sol, step.x_orig),
        orig_max_abs = maximum(abs.(x_sol .- step.x_orig)),
    )
end

function add_native_objective!(model::Model, x, step::SubP2StepModel.StepData)
    nxl, nul, nxi, nui, nq, _ = step.dims
    active_u = 1.0 - Float64(step.params["is_terminal"])
    rho_lx = Float64(step.params["rho_lx"])
    rho_lu = Float64(step.params["rho_lu"])
    rho_ix = Float64(step.params["rho_ix"])
    rho_iu = Float64(step.params["rho_iu"])

    xl_ideal = as_vec(step.params["xl_ideal"])
    ul_ideal = as_vec(step.params["ul_ideal"])
    y_xl = as_vec(step.params["y_xl"])
    y_ul = as_vec(step.params["y_ul"])
    xc_ideal = as_mat(step.params["xc_ideal"])
    uc_ideal = as_mat(step.params["uc_ideal"])
    y_xc = as_mat(step.params["y_xc"])
    y_uc = as_mat(step.params["y_uc"])

    @expression(model, obj_lx,
        0.5 * rho_lx * sum(
            (x[i] - xl_ideal[i] + y_xl[i] / (rho_lx + 1e-6))^2 for i in 1:nxl
        )
    )
    @expression(model, obj_lu,
        active_u * 0.5 * rho_lu * sum(
            (x[nxl + i] - ul_ideal[i] + y_ul[i] / (rho_lu + 1e-6))^2 for i in 1:nul
        )
    )
    base = nxl + nul
    stride = nxi + nui
    @expression(model, obj_xc,
        0.5 * rho_ix * sum(
            (
                x[base + (i - 1) * stride + j] -
                xc_ideal[i, j] +
                y_xc[i, j] / (rho_ix + 1e-6)
            )^2
            for i in 1:nq, j in 1:nxi
        )
    )
    @expression(model, obj_uc,
        active_u * 0.5 * rho_iu * sum(
            (
                x[base + (i - 1) * stride + nxi + j] -
                uc_ideal[i, j] +
                y_uc[i, j] / (rho_iu + 1e-6)
            )^2
            for i in 1:nq, j in 1:nui
        )
    )
    @objective(model, Min, obj_lx + obj_lu + obj_xc + obj_uc)
end

function build_jump_model_eq(step::SubP2StepModel.StepData; include_simple_ineq::Bool = true)
    model = Model()
    nxl, nul, nxi, nui, nq, _ = step.dims
    @variable(model, x[i = 1:length(step.x_init)], start = step.x_init[i])
    add_native_objective!(model, x, step)

    Pt = as_mat(step.params["Pt"])
    active_u = 1.0 - Float64(step.params["is_terminal"])
    pob1 = as_vec(step.params["pob1"])
    pob2 = as_vec(step.params["pob2"])
    ra = transpose(as_mat(step.params["ra"]))
    ro = Float64(step.params["ro"])
    rq = Float64(step.params["rq"])
    cl0 = Float64(step.params["cl0"])
    rl = Float64(step.params["rl"])
    t_min = Float64(step.params["t_min"])
    t_max = Float64(step.params["t_max"])
    ui_bound = Float64(step.params["ui_bound"])
    eps_margin = 1e-2
    num_dis = step.dims[6]
    g = Float64(step.params["g"])
    ml = Float64(step.params["ml"])
    mq = Float64(step.params["mq"])
    fmax = Float64(step.params["fmax"])
    Jl = as_mat(step.params["Jl"])
    Jl_inv = as_mat(step.params["Jl_inv"])

    q0 = x[7]; q1 = x[8]; q2 = x[9]; q3 = x[10]
    @expression(model, R11, 2 * (q0^2 + q1^2) - 1)
    @expression(model, R12, 2 * (q1 * q2 - q0 * q3))
    @expression(model, R13, 2 * (q1 * q3 + q0 * q2))
    @expression(model, R21, 2 * (q1 * q2 + q0 * q3))
    @expression(model, R22, 2 * (q0^2 + q2^2) - 1)
    @expression(model, R23, 2 * (q2 * q3 - q0 * q1))
    @expression(model, R31, 2 * (q1 * q3 - q0 * q2))
    @expression(model, R32, 2 * (q2 * q3 + q0 * q1))
    @expression(model, R33, 2 * (q0^2 + q3^2) - 1)
    Rl = Any[
        R11 R12 R13;
        R21 R22 R23;
        R31 R32 R33;
    ]

    @constraint(model, q0^2 + q1^2 + q2^2 + q3^2 == 1.0)

    base = nxl + nul
    stride = nxi + nui
    for i in 1:nq
        off = base + (i - 1) * stride
        d1 = x[off + 1]
        d2 = x[off + 2]
        d3 = x[off + 3]
        @constraint(model, d1^2 + d2^2 + d3^2 == 1.0)
    end

    @expression(model, fi_b1[i = 1:nq],
        R11 * (x[base + (i - 1) * stride + 1] * x[base + (i - 1) * stride + 13]) +
        R21 * (x[base + (i - 1) * stride + 2] * x[base + (i - 1) * stride + 13]) +
        R31 * (x[base + (i - 1) * stride + 3] * x[base + (i - 1) * stride + 13])
    )
    @expression(model, fi_b2[i = 1:nq],
        R12 * (x[base + (i - 1) * stride + 1] * x[base + (i - 1) * stride + 13]) +
        R22 * (x[base + (i - 1) * stride + 2] * x[base + (i - 1) * stride + 13]) +
        R32 * (x[base + (i - 1) * stride + 3] * x[base + (i - 1) * stride + 13])
    )
    @expression(model, fi_b3[i = 1:nq],
        R13 * (x[base + (i - 1) * stride + 1] * x[base + (i - 1) * stride + 13]) +
        R23 * (x[base + (i - 1) * stride + 2] * x[base + (i - 1) * stride + 13]) +
        R33 * (x[base + (i - 1) * stride + 3] * x[base + (i - 1) * stride + 13])
    )

    for r in 1:6
        expr = 0.0
        col = 1
        for i in 1:nq
            expr += Pt[r, col] * fi_b1[i]; col += 1
            expr += Pt[r, col] * fi_b2[i]; col += 1
            expr += Pt[r, col] * fi_b3[i]; col += 1
        end
        target =
            r == 1 ? (R11 * x[nxl + 1] + R21 * x[nxl + 2] + R31 * x[nxl + 3]) :
            r == 2 ? (R12 * x[nxl + 1] + R22 * x[nxl + 2] + R32 * x[nxl + 3]) :
            r == 3 ? (R13 * x[nxl + 1] + R23 * x[nxl + 2] + R33 * x[nxl + 3]) :
            x[nxl + r]
        @constraint(model, active_u * (expr - target) == 0.0)
    end

    if include_simple_ineq
        safe_r_l = ro + 0.5 * rq
        @constraint(model, safe_r_l^2 + eps_margin - ((x[1] - pob1[1])^2 + (x[2] - pob1[2])^2) <= 0.0)
        @constraint(model, safe_r_l^2 + eps_margin - ((x[1] - pob2[1])^2 + (x[2] - pob2[2])^2) <= 0.0)

        safe_r_q = ro + 2.0 * rq
        @expression(model, pi1[i = 1:nq],
            x[1] + (R11 * ra[i, 1] + R12 * ra[i, 2] + R13 * ra[i, 3]) + cl0 * x[base + (i - 1) * stride + 1]
        )
        @expression(model, pi2[i = 1:nq],
            x[2] + (R21 * ra[i, 1] + R22 * ra[i, 2] + R23 * ra[i, 3]) + cl0 * x[base + (i - 1) * stride + 2]
        )
        @constraint(model, [i = 1:nq], safe_r_q^2 + eps_margin - ((pi1[i] - pob1[1])^2 + (pi2[i] - pob1[2])^2) <= 0.0)
        @constraint(model, [i = 1:nq], safe_r_q^2 + eps_margin - ((pi1[i] - pob2[1])^2 + (pi2[i] - pob2[2])^2) <= 0.0)

        @constraint(model, [i = 1:nq], t_min - x[base + (i - 1) * stride + 13] <= 0.0)
        @constraint(model, [i = 1:nq], x[base + (i - 1) * stride + 13] - t_max <= 0.0)

        for i in 1:nq
            for j in 1:nui
                expr_u = active_u * (x[base + (i - 1) * stride + nxi + j] - ui_bound)
                expr_l = active_u * (-x[base + (i - 1) * stride + nxi + j] - ui_bound)
                @constraint(model, expr_u <= 0.0)
                @constraint(model, expr_l <= 0.0)
            end
        end

        @expression(model, pib1[kc = 1:num_dis, i = 1:nq],
            ra[i, 1] + (kc / num_dis) * cl0 * (R11 * x[base + (i - 1) * stride + 1] + R21 * x[base + (i - 1) * stride + 2] + R31 * x[base + (i - 1) * stride + 3])
        )
        @expression(model, pib2[kc = 1:num_dis, i = 1:nq],
            ra[i, 2] + (kc / num_dis) * cl0 * (R12 * x[base + (i - 1) * stride + 1] + R22 * x[base + (i - 1) * stride + 2] + R32 * x[base + (i - 1) * stride + 3])
        )

        for kc in 1:num_dis
            for i in 1:nq
                for j in i+1:nq
                    expr_pair = pair_term_indexed(x, base, stride, Rl, ra[i, 1], ra[i, 2], ra[i, 3], ra[j, 1], ra[j, 2], ra[j, 3], cl0, rq, eps_margin, kc, num_dis, i, j)
                    @constraint(
                        model,
                        expr_pair <= 0.0
                    )
                end
            end
        end

        ra_norm = [sqrt(ra[i, 1]^2 + ra[i, 2]^2 + ra[i, 3]^2) + 1e-9 for i in 1:nq]
        for i in 1:nq
            gio_u, gio_l = gio_terms_indexed(x, base, stride, Rl, ra[i, 1], ra[i, 2], ra[i, 3], cl0, rl, i)
            @constraint(model, gio_u <= 0.0)
            @constraint(model, gio_l <= 0.0)
        end

        Jw1 = Jl[1, 1] * x[11] + Jl[1, 2] * x[12] + Jl[1, 3] * x[13]
        Jw2 = Jl[2, 1] * x[11] + Jl[2, 2] * x[12] + Jl[2, 3] * x[13]
        Jw3 = Jl[3, 1] * x[11] + Jl[3, 2] * x[12] + Jl[3, 3] * x[13]
        cwJ1 = x[12] * Jw3 - x[13] * Jw2
        cwJ2 = x[13] * Jw1 - x[11] * Jw3
        cwJ3 = x[11] * Jw2 - x[12] * Jw1
        awl = Any[
            Jl_inv[1, 1] * (x[nxl + 4] - cwJ1) + Jl_inv[1, 2] * (x[nxl + 5] - cwJ2) + Jl_inv[1, 3] * (x[nxl + 6] - cwJ3),
            Jl_inv[2, 1] * (x[nxl + 4] - cwJ1) + Jl_inv[2, 2] * (x[nxl + 5] - cwJ2) + Jl_inv[2, 3] * (x[nxl + 6] - cwJ3),
            Jl_inv[3, 1] * (x[nxl + 4] - cwJ1) + Jl_inv[3, 2] * (x[nxl + 5] - cwJ2) + Jl_inv[3, 3] * (x[nxl + 6] - cwJ3),
        ]
        wl_vec = Any[x[11], x[12], x[13]]
        ul1 = x[nxl + 1]
        ul2 = x[nxl + 2]
        ul3 = x[nxl + 3]
        for i in 1:nq
            f1, f2, f3 = thrust_terms_indexed(x, base, stride, Rl, wl_vec, awl, ra[i, 1], ra[i, 2], ra[i, 3], cl0, mq, ml, g, ul1, ul2, ul3, i)
            expr_up = active_u * (f1^2 + f2^2 + f3^2 - fmax^2)
            expr_lo = active_u * (eps_margin - (f1^2 + f2^2 + f3^2))
            @constraint(model, expr_up <= 0.0)
            @constraint(model, expr_lo <= 0.0)
        end
    end

    return model, x
end

function solve_step_data(step::SubP2StepModel.StepData; print_level = MadNLP.ERROR, include_simple_ineq = true, max_iter = 200, acceptable_tol = 1e-4, acceptable_iter = 5, tol = 1e-8)
    t_build0 = time_ns()
    model, x = build_jump_model_eq(step; include_simple_ineq = include_simple_ineq)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    build_ms = (time_ns() - t_build0) / 1e6
    t0 = time_ns()
    result = madnlp(
        nlp;
        print_level = print_level,
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    solve_ms = (time_ns() - t0) / 1e6
    x_sol = vec(Float64.(result.solution))
    metrics = summarize_solution(step, x_sol)
    out = Dict(
        "status" => string(result.status),
        "iter" => result.iter,
        "build_ms" => build_ms,
        "solve_ms" => solve_ms,
        "wall_ms" => build_ms + solve_ms,
        "objective" => metrics.objective,
        "eq_inf" => metrics.eq_inf,
        "ineq_vio" => metrics.ineq_vio,
        "orig_rmse" => metrics.orig_rmse,
        "orig_max_abs" => metrics.orig_max_abs,
        "include_simple_ineq" => include_simple_ineq,
        "x_sol" => x_sol,
    )
    println("MadNLP JuMP native status: ", out["status"])
    println("iter: ", out["iter"])
    println("build_ms: ", round(out["build_ms"]; digits = 3))
    println("solve_ms: ", round(out["solve_ms"]; digits = 3))
    println("wall_ms: ", round(out["wall_ms"]; digits = 3))
    println("eq_inf: ", out["eq_inf"])
    println("ineq_vio: ", out["ineq_vio"])
    println("orig_rmse: ", out["orig_rmse"])
    println("orig_max_abs: ", out["orig_max_abs"])
    return out
end

function solve_step(path::String; print_level = MadNLP.ERROR, include_simple_ineq = true, max_iter = 200, acceptable_tol = 1e-4, acceptable_iter = 5, tol = 1e-8)
    step = load_step_data(path)
    return solve_step_data(
        step;
        print_level = print_level,
        include_simple_ineq = include_simple_ineq,
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
end

function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/solve_step_madnlp_jump_native_eq.jl <snapshot.json> [out.json] [--eq-only] [--max-iter=N] [--acceptable-tol=X] [--acceptable-iter=N] [--tol=X]")
    end
    out_path = nothing
    include_simple_ineq = true
    max_iter = 200
    acceptable_tol = 1e-4
    acceptable_iter = 5
    tol = 1e-8
    for arg in ARGS[2:end]
        if arg == "--eq-only"
            include_simple_ineq = false
        elseif startswith(arg, "--max-iter=")
            max_iter = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--acceptable-tol=")
            acceptable_tol = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--acceptable-iter=")
            acceptable_iter = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=", limit = 2)[2])
        else
            out_path = arg
        end
    end
    result = solve_step(
        ARGS[1];
        include_simple_ineq = include_simple_ineq,
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    if out_path !== nothing
        open(out_path, "w") do io
            JSON3.pretty(io, result)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
