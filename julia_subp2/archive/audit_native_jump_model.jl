import Pkg
Pkg.activate(@__DIR__)

using JSON3
using JuMP
using MadNLP
using NLPModels
using NLPModelsJuMP

include("step_model.jl")
using .SubP2StepModel

module NativeJumpStep
include("solve_step_madnlp_jump_native_eq.jl")
end

function vecf(x)
    return Float64.(x)
end

function audit_point(name::String, nlp, step, x::Vector{Float64})
    obj_model = NLPModels.obj(nlp, x)
    cons_model = vecf(NLPModels.cons(nlp, x))
    eq_ref = vecf(NativeJumpStep.SubP2StepModel.equality_residual(x, step.params, step.dims))
    ineq_ref = vecf(NativeJumpStep.SubP2StepModel.inequality_residual(x, step.params, step.dims))
    cons_ref = vcat(eq_ref, ineq_ref)

    eq_dim = length(eq_ref)
    ineq_dim = length(ineq_ref)
    @assert length(cons_model) == length(cons_ref)

    obj_ref = NativeJumpStep.SubP2StepModel.objective(x, step.params, step.dims)
    obj_diff = abs(obj_model - obj_ref)
    cons_diff = abs.(cons_model .- cons_ref)
    eq_max = maximum(cons_diff[1:eq_dim])
    ineq_max = maximum(cons_diff[eq_dim+1:eq_dim+ineq_dim])
    all_max = maximum(cons_diff)

    out = Dict(
        "name" => name,
        "objective_model" => obj_model,
        "objective_ref" => obj_ref,
        "objective_abs_diff" => obj_diff,
        "eq_dim" => eq_dim,
        "ineq_dim" => ineq_dim,
        "constraint_abs_diff_max" => all_max,
        "eq_abs_diff_max" => eq_max,
        "ineq_abs_diff_max" => ineq_max,
    )
    println("[audit] point=", name)
    println("  objective_abs_diff = ", obj_diff)
    println("  eq_abs_diff_max    = ", eq_max)
    println("  ineq_abs_diff_max  = ", ineq_max)
    println("  all_abs_diff_max   = ", all_max)
    return out
end

function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/audit_native_jump_model.jl <snapshot.json> [out.json]")
    end
    path = ARGS[1]
    out_path = length(ARGS) >= 2 ? ARGS[2] : nothing

    step = NativeJumpStep.SubP2StepModel.load_step_data(path)
    model, _ = NativeJumpStep.build_jump_model_eq(step; include_simple_ineq = true)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)

    x_init = vecf(step.x_init)
    x_orig = vecf(step.x_orig)

    init_out = audit_point("x_init", nlp, step, x_init)
    orig_out = audit_point("x_orig", nlp, step, x_orig)

    result = Dict(
        "snapshot" => path,
        "x_init" => init_out,
        "x_orig" => orig_out,
    )
    if out_path !== nothing
        open(out_path, "w") do io
            JSON3.pretty(io, result)
        end
    end
end

main()
