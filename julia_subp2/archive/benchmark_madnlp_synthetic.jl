if !isdefined(Main, :__SUBP2_JULIA_ENV_ACTIVE__)
    import Pkg
    Pkg.activate(@__DIR__)
    const __SUBP2_JULIA_ENV_ACTIVE__ = true
end

using JSON3
using JuMP
using MadNLP
using NLPModelsJuMP

function load_cfg(path::String)
    return JSON3.read(read(path, String))
end

function json_matrix(a)
    m = length(a)
    n = length(a[1])
    out = Matrix{Float64}(undef, m, n)
    for i in 1:m, j in 1:n
        out[i, j] = Float64(a[i][j])
    end
    return out
end

function json_vector(a)
    return [Float64(v) for v in a]
end

function build_model(cfg)
    n = Int(cfg["n_agents"])
    wx = Float64(cfg["wx"])
    wu = Float64(cfg["wu"])
    wc = Float64(cfg["wc"])
    min_sep = Float64(cfg["min_sep"])
    u_bound = Float64(cfg["u_bound"])
    thrust_max = Float64(cfg["thrust_max"])
    thrust_min2 = Float64(cfg["thrust_min2"])
    safe_r = Float64(cfg["safe_r"])
    p_ref = json_matrix(cfg["p_ref"])
    u_ref = json_matrix(cfg["u_ref"])
    x0 = json_vector(cfg["x0"])
    obs = json_matrix(cfg["obstacles"])
    target_u = json_vector(cfg["target_u"])

    model = Model()
    @variable(model, p[i = 1:n, j = 1:3], start = x0[(i - 1) * 6 + j])
    @variable(model, u[i = 1:n, j = 1:3], start = x0[(i - 1) * 6 + 3 + j])

    @objective(
        model,
        Min,
        sum(
            0.5 * wx * sum((p[i, j] - p_ref[i, j])^2 for j in 1:3) +
            0.5 * wu * sum((u[i, j] - u_ref[i, j])^2 for j in 1:3) +
            wc * (sum(p[i, j] * u[i, j] for j in 1:3))^2
            for i in 1:n
        )
    )

    @constraint(model, [i = 1:n], p[i, 1]^2 + p[i, 2]^2 + p[i, 3]^2 == 1.0)
    @constraint(model, [j = 1:3], sum(u[i, j] for i in 1:n) == target_u[j])

    nobs = size(obs, 1)
    @constraint(
        model,
        [i = 1:n, k = 1:nobs],
        safe_r^2 - ((p[i, 1] - obs[k, 1])^2 + (p[i, 2] - obs[k, 2])^2) <= 0.0
    )
    @constraint(
        model,
        [i = 1:n, j = i+1:n],
        min_sep^2 - (
            (p[i, 1] - p[j, 1])^2 +
            (p[i, 2] - p[j, 2])^2 +
            (p[i, 3] - p[j, 3])^2
        ) <= 0.0
    )
    @constraint(model, [i = 1:n, j = 1:3], u[i, j] - u_bound <= 0.0)
    @constraint(model, [i = 1:n, j = 1:3], -u[i, j] - u_bound <= 0.0)
    @constraint(model, [i = 1:n], u[i, 1]^2 + u[i, 2]^2 + u[i, 3]^2 - thrust_max^2 <= 0.0)
    @constraint(model, [i = 1:n], thrust_min2 - (u[i, 1]^2 + u[i, 2]^2 + u[i, 3]^2) <= 0.0)

    return model
end

function solve_once(cfg; max_iter = 200, tol = 1e-8, acceptable_tol = 1e-6, acceptable_iter = 5)
    t_build = time_ns()
    model = build_model(cfg)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    build_ms = (time_ns() - t_build) / 1e6

    t_solve = time_ns()
    result = madnlp(
        nlp;
        print_level = MadNLP.ERROR,
        max_iter = max_iter,
        tol = tol,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
    )
    solve_ms = (time_ns() - t_solve) / 1e6
    cval = result.constraints
    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon
    eq_mask = isfinite.(lcon) .& isfinite.(ucon) .& (lcon .== ucon)
    eq_inf = any(eq_mask) ? maximum(abs.(cval[eq_mask] .- lcon[eq_mask])) : 0.0
    lower_vio = max.(lcon .- cval, 0.0)
    upper_vio = max.(cval .- ucon, 0.0)
    ineq_vio = maximum(max.(lower_vio, upper_vio))
    return Dict(
        "status" => string(result.status),
        "iter" => result.iter,
        "build_ms" => build_ms,
        "solve_ms" => solve_ms,
        "wall_ms" => build_ms + solve_ms,
        "objective" => result.objective,
        "eq_inf" => eq_inf,
        "ineq_vio" => ineq_vio,
    )
end

function main()
    if length(ARGS) < 1
        error("usage: julia benchmark_madnlp_synthetic.jl <config.json> [out.json]")
    end
    cfg = load_cfg(ARGS[1])
    run1 = solve_once(cfg)
    run2 = solve_once(cfg)
    out = Dict("run1" => run1, "run2" => run2)
    println("MadNLP synthetic run1: ", run1)
    println("MadNLP synthetic run2: ", run2)
    if length(ARGS) >= 2
        open(ARGS[2], "w") do io
            JSON3.pretty(io, out)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
