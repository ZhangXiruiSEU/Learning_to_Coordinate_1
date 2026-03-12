import Pkg
const _ROOT = dirname(@__DIR__)
Pkg.activate(_ROOT)

include(joinpath(_ROOT, "step_model.jl"))
using .SubP2StepModel

function report_vec_diff(name, a, b)
    @assert length(a) == length(b)
    diffs = abs.(a .- b)
    idx = argmax(diffs)
    println(name, " max diff = ", diffs[idx], " at idx=", idx, " julia=", a[idx], " python=", b[idx])
end

function check_model(path::String)
    step = load_step_data(path)

    obj_init = objective(step.x_init, step.params, step.dims)
    eq_init = equality_residual(step.x_init, step.params, step.dims)
    ineq_init = inequality_residual(step.x_init, step.params, step.dims)
    obj_orig = objective(step.x_orig, step.params, step.dims)
    eq_orig = equality_residual(step.x_orig, step.params, step.dims)
    ineq_orig = inequality_residual(step.x_orig, step.params, step.dims)

    println("objective init max diff = ", abs(obj_init - step.init_obj_ref))
    report_vec_diff("equality init", Float64.(eq_init), step.init_eq_ref)
    report_vec_diff("ineq init", Float64.(ineq_init), step.init_ineq_ref)
    println("objective orig max diff = ", abs(obj_orig - step.orig_obj_ref))
    report_vec_diff("equality orig", Float64.(eq_orig), step.orig_eq_ref)
    report_vec_diff("ineq orig", Float64.(ineq_orig), step.orig_ineq_ref)
end

function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/archive/validate_step_model.jl <snapshot.json>")
    end
    check_model(ARGS[1])
end

main()
