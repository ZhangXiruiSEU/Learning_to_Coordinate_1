import Pkg
Pkg.activate(dirname(@__DIR__))

using JSON3
using CUDA
using ExaModels
using MadNLP

function as_vec(x)
    return [Float64(v) for v in x]
end

function as_mat(x)
    rows = length(x)
    cols = length(x[1])
    out = Matrix{Float64}(undef, rows, cols)
    for i in 1:rows
        @assert length(x[i]) == cols
        for j in 1:cols
            out[i, j] = Float64(x[i][j])
        end
    end
    return out
end

function check_snapshot(path::String)
    data = JSON3.read(read(path, String))

    meta = data["meta"]
    step = data["step"]
    para2 = data["para2_contract"]

    nxl = Int(meta["nxl"])
    nul = Int(meta["nul"])
    nxi = Int(meta["nxi"])
    nui = Int(meta["nui"])
    nq = Int(meta["nq"])
    decision_dim = Int(meta["decision_dim"])

    x_init = as_vec(step["x_init"])
    x_orig = as_vec(step["original_step_reference"])

    @assert length(x_init) == decision_dim
    @assert length(x_orig) == decision_dim
    @assert length(as_vec(step["params_t"]["xl_ideal"])) == nxl
    @assert length(as_vec(step["params_t"]["ul_ideal"])) == nul
    @assert size(as_mat(step["params_t"]["xc_ideal"])) == (nq, nxi)
    @assert size(as_mat(step["params_t"]["uc_ideal"])) == (nq, nui)
    @assert length(as_vec(para2["para_l"])) > 0
    @assert length(as_vec(para2["para_i"])) > 0

    println("Julia version: ", VERSION)
    println("CUDA.functional(): ", CUDA.functional())
    println("Loaded packages: ExaModels, JSON3, MadNLP")
    println("Snapshot contract OK")
    println("task=", meta["task_idx"], " horizon=", meta["horizon"],
            " admm_iter=", meta["target_admm_iter"], " step=", meta["target_step"])
    println("decision_dim=", decision_dim,
            " eq_dim=", Int(meta["eq_dim"]),
            " ineq_dim=", Int(meta["ineq_dim"]))
    println("init eq_inf=", step["init_metrics"]["eq_inf"],
            " init ineq_vio=", step["init_metrics"]["ineq_vio"])
    println("orig eq_inf=", step["original_metrics"]["eq_inf"],
            " orig ineq_vio=", step["original_metrics"]["ineq_vio"])
    println()
    println("Next implementation step:")
    println("- rebuild objective and constraints in ExaModels using this exact contract")
    println("- keep x ordering identical to Python/JAX: [xl, ul, (xi,ui)_1, ... , (xi,ui)_nq]")
    println("- validate eq/ineq values against Python before benchmarking MadNLP")
end

function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/archive/load_snapshot.jl <snapshot.json>")
    end
    check_snapshot(ARGS[1])
end

main()
