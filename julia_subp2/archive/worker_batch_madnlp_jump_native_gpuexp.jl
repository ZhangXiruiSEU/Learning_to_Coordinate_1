if !haskey(ENV, "JULIA_SUBP2_LINEAR_SOLVER")
    ENV["JULIA_SUBP2_LINEAR_SOLVER"] = "lapackcuda"
end
if !haskey(ENV, "JULIA_SUBP2_KKT_SYSTEM")
    ENV["JULIA_SUBP2_KKT_SYSTEM"] = "dense_condensed"
end
if !haskey(ENV, "JULIA_SUBP2_CALLBACK")
    ENV["JULIA_SUBP2_CALLBACK"] = "dense"
end
if !haskey(ENV, "JULIA_SUBP2_THREAD_SCHEDULE")
    ENV["JULIA_SUBP2_THREAD_SCHEDULE"] = "default"
end

include(joinpath(dirname(@__DIR__), "worker_batch_madnlp_jump_native.jl"))
