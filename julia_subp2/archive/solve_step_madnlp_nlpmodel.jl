import Pkg
Pkg.activate(@__DIR__)

using JSON3
using MadNLP
using NLPModels

include("nlpmodel_step.jl")

function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/solve_step_madnlp_nlpmodel.jl <snapshot.json> [out.json]")
    end
    out_path = length(ARGS) >= 2 ? ARGS[2] : nothing
    result = solve_nohessian_step(ARGS[1])
    println("MadNLP NLPModel status: ", result["status"])
    println("iter: ", result["iter"])
    println("wall_ms: ", round(result["wall_ms"]; digits = 3))
    println("eq_inf: ", result["eq_inf"])
    println("ineq_vio: ", result["ineq_vio"])
    println("orig_rmse: ", result["orig_rmse"])
    println("orig_max_abs: ", result["orig_max_abs"])
    if out_path !== nothing
        open(out_path, "w") do io
            JSON3.pretty(io, result)
        end
    end
end

main()
