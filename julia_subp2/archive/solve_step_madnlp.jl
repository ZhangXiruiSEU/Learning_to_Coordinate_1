import Pkg
Pkg.activate(@__DIR__)

using JSON3
using LinearAlgebra
using MadNLP
using ExaModels

include("step_model.jl")
using .SubP2StepModel

function rmse(a::AbstractVector, b::AbstractVector)
    return sqrt(sum(abs2, a .- b) / length(a))
end

function summarize_solution(step::StepData, x_sol::Vector{Float64})
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

function solve_step(
    path::String;
    print_level = MadNLP.ERROR,
    include_ineq = true,
    include_thrust = true,
    callback = nothing,
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
)
    step = load_step_data(path)
    model, x = build_examodel(step; include_ineq = include_ineq, include_thrust = include_thrust)

    t0 = time_ns()
    kwargs = (
        print_level = print_level,
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    result = isnothing(callback) ?
        madnlp(model; kwargs...) :
        madnlp(model; kwargs..., callback = callback)
    wall_ms = (time_ns() - t0) / 1e6

    x_sol = vec(Float64.(solution(result, x)))
    metrics = summarize_solution(step, x_sol)

    out = Dict(
        "status" => string(result.status),
        "iter" => (hasproperty(result, :iter) ? getproperty(result, :iter) : missing),
        "wall_ms" => wall_ms,
        "objective" => metrics.objective,
        "eq_inf" => metrics.eq_inf,
        "ineq_vio" => metrics.ineq_vio,
        "orig_rmse" => metrics.orig_rmse,
        "orig_max_abs" => metrics.orig_max_abs,
        "include_ineq" => include_ineq,
        "include_thrust" => include_thrust,
        "callback" => isnothing(callback) ? "default" : string(callback),
        "max_iter" => max_iter,
        "acceptable_tol" => acceptable_tol,
        "acceptable_iter" => acceptable_iter,
        "tol" => tol,
    )

    println("MadNLP status: ", out["status"])
    println("iter: ", out["iter"])
    println("wall_ms: ", round(out["wall_ms"]; digits = 3))
    println("eq_inf: ", out["eq_inf"])
    println("ineq_vio: ", out["ineq_vio"])
    println("orig_rmse: ", out["orig_rmse"])
    println("orig_max_abs: ", out["orig_max_abs"])

    return out
end

function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/solve_step_madnlp.jl <snapshot.json> [out.json] [--eq-only]")
    end
    out_path = nothing
    include_ineq = true
    include_thrust = true
    callback = nothing
    max_iter = 200
    acceptable_tol = 1e-4
    acceptable_iter = 5
    tol = 1e-8
    for arg in ARGS[2:end]
        if arg == "--eq-only"
            include_ineq = false
        elseif arg == "--skip-thrust"
            include_thrust = false
        elseif arg == "--sparse-callback"
            callback = MadNLP.SparseCallback
        elseif arg == "--dense-callback"
            callback = MadNLP.DenseCallback
        elseif startswith(arg, "--max-iter=")
            max_iter = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--acceptable-tol=")
            acceptable_tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--acceptable-iter=")
            acceptable_iter = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        else
            out_path = arg
        end
    end
    result = solve_step(
        ARGS[1];
        include_ineq = include_ineq,
        include_thrust = include_thrust,
        callback = callback,
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

main()
