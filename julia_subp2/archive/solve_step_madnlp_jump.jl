import Pkg
Pkg.activate(@__DIR__)

using JSON3
using JuMP
using MadNLP
using NLPModelsJuMP

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

function build_jump_model(step::StepData; include_ineq::Bool = true)
    n = length(step.x_init)
    meq = length(step.init_eq_ref)
    mineq = length(step.init_ineq_ref)

    model = Model()
    @variable(model, x[i = 1:n], start = step.x_init[i])

    obj_fun(args...) = SubP2StepModel.objective(collect(args), step.params, step.dims)
    register(model, :subp2_obj, n, obj_fun; autodiff = true)
    @NLobjective(model, Min, subp2_obj(x...))

    eq_syms = Symbol[]
    for i in 1:meq
        sym = Symbol("subp2_eq_", i)
        push!(eq_syms, sym)
        eq_fun(args...) = SubP2StepModel.equality_component(collect(args), step.params, step.dims, i)
        register(model, sym, n, eq_fun; autodiff = true)
    end
    for i in 1:meq
        sym = eq_syms[i]
        add_nonlinear_constraint(model, :($(sym)($(x...)) == 0.0))
    end

    if include_ineq
        ineq_syms = Symbol[]
        for i in 1:mineq
            sym = Symbol("subp2_ineq_", i)
            push!(ineq_syms, sym)
            ineq_fun(args...) = SubP2StepModel.inequality_component(collect(args), step.params, step.dims, i)
            register(model, sym, n, ineq_fun; autodiff = true)
        end
        for i in 1:mineq
            sym = ineq_syms[i]
            add_nonlinear_constraint(model, :($(sym)($(x...)) <= 0.0))
        end
    end

    return model, x
end

function solve_step(
    path::String;
    print_level = MadNLP.ERROR,
    include_ineq = true,
    hessian_approximation = MadNLP.CompactLBFGS,
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
)
    step = load_step_data(path)
    model, x = build_jump_model(step; include_ineq = include_ineq)
    nlp = NLPModelsJuMP.MathOptNLPModel(model; hessian = false)

    t0 = time_ns()
    result = madnlp(
        nlp;
        print_level = print_level,
        hessian_approximation = hessian_approximation,
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    wall_ms = (time_ns() - t0) / 1e6

    x_sol = vec(Float64.(result.solution))
    metrics = summarize_solution(step, x_sol)
    out = Dict(
        "status" => string(result.status),
        "iter" => result.iter,
        "wall_ms" => wall_ms,
        "objective" => metrics.objective,
        "eq_inf" => metrics.eq_inf,
        "ineq_vio" => metrics.ineq_vio,
        "orig_rmse" => metrics.orig_rmse,
        "orig_max_abs" => metrics.orig_max_abs,
        "include_ineq" => include_ineq,
        "max_iter" => max_iter,
        "acceptable_tol" => acceptable_tol,
        "acceptable_iter" => acceptable_iter,
        "tol" => tol,
    )

    println("MadNLP JuMP status: ", out["status"])
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
        error("usage: julia julia_subp2/solve_step_madnlp_jump.jl <snapshot.json> [out.json] [--eq-only]")
    end
    out_path = nothing
    include_ineq = true
    max_iter = 200
    acceptable_tol = 1e-4
    acceptable_iter = 5
    tol = 1e-8
    for arg in ARGS[2:end]
        if arg == "--eq-only"
            include_ineq = false
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
