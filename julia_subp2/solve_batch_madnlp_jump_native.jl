import Pkg
Pkg.activate(@__DIR__)
const __SUBP2_JULIA_ENV_ACTIVE__ = true

using JSON3
using MadNLP
using JuMP
using Base.Threads
const MOI = JuMP.MOI
const _CUDA_AVAILABLE = try
    @eval import CUDA
    true
catch
    false
end
const _MADNLPGPU_AVAILABLE = try
    @eval import MadNLPGPU
    true
catch
    false
end

include("step_model.jl")
using .SubP2StepModel
include("solve_step_madnlp_jump_native_eq.jl")

# -----------------------------------------------------------------------------
# 这个文件是 Julia 侧 SubP2 的总调度中心。
#
# 当前主线真正走的是：
#   Python -> persistent worker -> solve_batch_payload(data; result_format=:stacked...)
#   -> stacked runtime template cache -> Threads.@threads 并行解每个 step
#
# 因此，这里同时承担了四类职责：
# 1. backend / 线程调度配置
# 2. 单步 JuMP/MadNLP 模型缓存
# 3. stacked batch 参数更新与并行求解
# 4. CLI / 文件输入兼容
#
# 当前主线的必经调用链可以直接按这个顺序追：
# solve_batch_payload (stacked path)
#   -> _ensure_stacked_batch_template!
#   -> solve_one_stacked(idx)
#   -> _solve_stacked_cached_step!
#   -> update_parameterized_step_from_stacked!
#   -> solve_cached_step_compact!
# 如果你现在只是想读懂“当前正式主线到底怎么跑”，最小阅读顺序是：
# 1. solve_batch_payload(...)               # 看 stacked path，而不是 legacy path
# 2. _ensure_stacked_batch_template!(...)   # 看模板/缓存是怎么准备的
# 3. _solve_stacked_cached_step!(...)       # 看单个 step 在 batch 中如何被执行
# 4. update_parameterized_step_from_stacked!(...)
#                                          # 看每轮 ADMM / 每个 step 真正更新了哪些参数
# 5. solve_cached_step_compact!(...)        # 看单步 solve + warm start + 紧凑回包
# 6. build_cached_step_solver(...)          # 看缓存槽位首次如何构建
# 7. build_jump_model_eq_parameterized(...) # 最后再看单步 NLP 模型本体
#
# 可以先跳过的内容：
# - update_parameterized_step!(...)         # legacy JSON step 路径
# - prepare_batch_payload!(...)             # 预热/诊断入口
# - solve_batch(path) / main()              # CLI / 文件模式
# - GPU outer serial 分支                   # 当前 CPU 主线默认不走
# -----------------------------------------------------------------------------

# 每个 step 对应一个可复用的 solver 槽位。
# 缓存里除了 model 本身，还保存：
# - 参数引用（便于 set_parameter_value）
# - warm start 解向量
# - 最近一次 solver 选项
mutable struct CachedStepSolver
    step::SubP2StepModel.StepData
    model
    xrefs
    params
    backend_key::String
    warm_slot_id::Int
    warm_solution::Union{Nothing, Vector{Float64}}
    last_max_iter::Int
    last_acceptable_tol::Float64
    last_acceptable_iter::Int
    last_tol::Float64
end

const _STEP_SOLVER_CACHE = Dict{Tuple{Int, String}, CachedStepSolver}()
const _STEP_SOLVER_CACHE_LOCK = ReentrantLock()
const _STACKED_BATCH_TEMPLATE_CACHE = Dict{Any, Vector{CachedStepSolver}}()
const _STACKED_BATCH_TEMPLATE_CACHE_LOCK = ReentrantLock()

# 结构缓存键只区分"问题结构"，不区分当前轮次的参数值。
function structure_cache_key(step::SubP2StepModel.StepData)
    return string(step.dims, "|ineq=true")
end

function structure_cache_key(dims::NTuple{6, Int})
    return string(dims, "|ineq=true")
end

function _get_backend_config()
    linear_solver_name = lowercase(strip(get(ENV, "JULIA_SUBP2_LINEAR_SOLVER", "mumps")))
    kkt_system_name = lowercase(strip(get(ENV, "JULIA_SUBP2_KKT_SYSTEM", "default")))
    callback_name = lowercase(strip(get(ENV, "JULIA_SUBP2_CALLBACK", "default")))
    return (
        linear_solver_name = linear_solver_name,
        kkt_system_name = kkt_system_name,
        callback_name = callback_name,
    )
end

function _backend_key()
    cfg = _get_backend_config()
    return string(
        "ls=", cfg.linear_solver_name,
        "|kkt=", cfg.kkt_system_name,
        "|cb=", cfg.callback_name,
    )
end

function _gpu_linear_solver_enabled()
    cfg = _get_backend_config()
    return cfg.linear_solver_name in ("lapackcuda", "cudss")
end

# GPU 外层模式只影响"step 级并行"怎么分发，不改变单步模型定义。
function _gpu_outer_mode()
    mode = lowercase(strip(get(ENV, "JULIA_SUBP2_GPU_OUTER_MODE", "serial")))
    mode in ("serial", "threaded") || error("unsupported JULIA_SUBP2_GPU_OUTER_MODE=$(mode)")
    return mode
end

function _thread_schedule_name()
    return lowercase(strip(get(ENV, "JULIA_SUBP2_THREAD_SCHEDULE", "default")))
end

# step 级并行的统一入口。
# 当前 CPU 主线就是把 batch 中的 n 个 step 丢给这里按线程调度。
function _run_threaded_range!(f, n::Int)
    sched = _thread_schedule_name()
    if sched == "dynamic"
        Threads.@threads :dynamic for idx in 1:n
            f(idx)
        end
    elseif sched == "static"
        Threads.@threads :static for idx in 1:n
            f(idx)
        end
    else
        Threads.@threads for idx in 1:n
            f(idx)
        end
    end
    return nothing
end

function _maybe_getfield(mod, name::Symbol)
    return isdefined(mod, name) ? getfield(mod, name) : nothing
end

function _resolve_linear_solver(name::String)
    return name == "mumps" ? MadNLP.MumpsSolver :
        name == "lapack" ? MadNLP.LapackCPUSolver :
        name == "umfpack" ? MadNLP.UmfpackSolver :
        name == "cholmod" ? MadNLP.CHOLMODSolver :
        name == "ldl" ? MadNLP.LDLSolver :
        (name == "cudss" && _MADNLPGPU_AVAILABLE) ? MadNLPGPU.CUDSSSolver :
        (name == "lapackcuda" && _MADNLPGPU_AVAILABLE) ? MadNLPGPU.LapackCUDASolver :
        error("unsupported JULIA_SUBP2_LINEAR_SOLVER=$(name)")
end

function _resolve_kkt_system(name::String)
    return name == "default" ? nothing :
        name == "dense" ? _maybe_getfield(MadNLP, :DenseKKTSystem) :
        name == "dense_condensed" ? _maybe_getfield(MadNLP, :DenseCondensedKKTSystem) :
        name == "sparse" ? _maybe_getfield(MadNLP, :SparseKKTSystem) :
        name == "sparse_condensed" ? _maybe_getfield(MadNLP, :SparseCondensedKKTSystem) :
        error("unsupported JULIA_SUBP2_KKT_SYSTEM=$(name)")
end

function _resolve_callback(name::String)
    return name == "default" ? nothing :
        name == "dense" ? _maybe_getfield(MadNLP, :DenseCallback) :
        error("unsupported JULIA_SUBP2_CALLBACK=$(name)")
end

function _configure_madnlp_backend!(model::Model)
    cfg = _get_backend_config()
    if cfg.linear_solver_name == "cudss"
        _CUDA_AVAILABLE || error("JULIA_SUBP2_LINEAR_SOLVER=cudss requires CUDA.jl to be available.")
        set_optimizer_attribute(model, "array_type", CUDA.CuArray)
    end
    set_optimizer_attribute(model, "linear_solver", _resolve_linear_solver(cfg.linear_solver_name))
    kkt_system = _resolve_kkt_system(cfg.kkt_system_name)
    callback = _resolve_callback(cfg.callback_name)
    if cfg.linear_solver_name == "lapackcuda"
        # MadNLPGPU's LAPACK CUDA backend is designed for dense callbacks/KKT systems.
        callback = something(callback, MadNLP.DenseCallback)
        kkt_system = something(kkt_system, MadNLP.DenseCondensedKKTSystem)
    end
    if kkt_system !== nothing
        set_optimizer_attribute(model, "kkt_system", kkt_system)
    end
    if callback !== nothing
        set_optimizer_attribute(model, "callback", callback)
    end
end

# 下面三个 helper 用 JuMP Parameter 构造"可更新参数"，
# 这样后面每轮 ADMM 只改参数，不重建模型。
function _build_param_scalar(model, name::Symbol, value::Float64)
    pref = @variable(model, base_name = string(name), set = Parameter(value))
    model[name] = pref
    return pref
end

function _build_param_vec(model, name::Symbol, values::AbstractVector{<:Real})
    p = @variable(model, [i = 1:length(values)], base_name = string(name), set = Parameter(Float64(values[i])))
    model[name] = p
    return p
end

function _build_param_mat(model, name::Symbol, values::AbstractMatrix{<:Real})
    nr, nc = size(values)
    p = @variable(
        model,
        [i = 1:nr, j = 1:nc],
        base_name = string(name),
        set = Parameter(Float64(values[i, j])),
    )
    model[name] = p
    return p
end

# 为单步 SubP2 构造参数化一致性目标。
# 对应 Python/ADMM 传过来的 ideal primal / dual / rho。
function add_native_objective_parameterized!(model::Model, x, dims)
    nxl, nul, nxi, nui, nq, _ = dims

    rho_lx = _build_param_scalar(model, :rho_lx_param, 1.0)
    rho_lu = _build_param_scalar(model, :rho_lu_param, 1.0)
    rho_ix = _build_param_scalar(model, :rho_ix_param, 1.0)
    rho_iu = _build_param_scalar(model, :rho_iu_param, 1.0)
    is_terminal = _build_param_scalar(model, :is_terminal_param, 0.0)
    xl_ideal = _build_param_vec(model, :xl_ideal_param, zeros(nxl))
    ul_ideal = _build_param_vec(model, :ul_ideal_param, zeros(nul))
    y_xl = _build_param_vec(model, :y_xl_param, zeros(nxl))
    y_ul = _build_param_vec(model, :y_ul_param, zeros(nul))
    xc_ideal = _build_param_mat(model, :xc_ideal_param, zeros(nq, nxi))
    uc_ideal = _build_param_mat(model, :uc_ideal_param, zeros(nq, nui))
    y_xc = _build_param_mat(model, :y_xc_param, zeros(nq, nxi))
    y_uc = _build_param_mat(model, :y_uc_param, zeros(nq, nui))

    active_u = 1.0 - is_terminal

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

    return (
        rho_lx = rho_lx,
        rho_lu = rho_lu,
        rho_ix = rho_ix,
        rho_iu = rho_iu,
        is_terminal = is_terminal,
        xl_ideal = xl_ideal,
        ul_ideal = ul_ideal,
        y_xl = y_xl,
        y_ul = y_ul,
        xc_ideal = xc_ideal,
        uc_ideal = uc_ideal,
        y_xc = y_xc,
        y_uc = y_uc,
    )
end

# 构造单步 SubP2 的 JuMP/MadNLP 模型。
#
# 决策变量 x 的布局是：
# - 负载状态 xl
# - 负载控制 ul
# - 每架无人机对应的缆绳状态/控制块
#
# 这里建的是"结构固定、参数可更新"的模型骨架。
function build_jump_model_eq_parameterized(dims; include_simple_ineq::Bool = true)
    model = Model(MadNLP.Optimizer)
    set_silent(model)
    _configure_madnlp_backend!(model)
    nxl, nul, nxi, nui, nq, num_dis = dims
    @variable(model, x[i = 1:(nxl + nul + nq * (nxi + nui))], start = 0.0)
    obj_params = add_native_objective_parameterized!(model, x, dims)

    Pt = _build_param_mat(model, :Pt_param, zeros(6, 3 * nq))
    pob1 = _build_param_vec(model, :pob1_param, zeros(3))
    pob2 = _build_param_vec(model, :pob2_param, zeros(3))
    ra = _build_param_mat(model, :ra_param, zeros(nq, 3))
    ro = _build_param_scalar(model, :ro_param, 0.0)
    rq = _build_param_scalar(model, :rq_param, 0.0)
    cl0 = _build_param_scalar(model, :cl0_param, 0.0)
    rl = _build_param_scalar(model, :rl_param, 0.0)
    t_min = _build_param_scalar(model, :t_min_param, 0.0)
    t_max = _build_param_scalar(model, :t_max_param, 0.0)
    ui_bound = _build_param_scalar(model, :ui_bound_param, 0.0)
    g = _build_param_scalar(model, :g_param, 0.0)
    ml = _build_param_scalar(model, :ml_param, 0.0)
    mq = _build_param_scalar(model, :mq_param, 0.0)
    fmax = _build_param_scalar(model, :fmax_param, 0.0)
    Jl = _build_param_mat(model, :Jl_param, zeros(3, 3))
    Jl_inv = _build_param_mat(model, :Jl_inv_param, zeros(3, 3))

    active_u = 1.0 - obj_params.is_terminal
    eps_margin = 1e-2

    # 由负载四元数显式展开旋转矩阵，后续多处约束都会复用。
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

    # 负载四元数单位模约束。
    @constraint(model, q0^2 + q1^2 + q2^2 + q3^2 == 1.0)

    base = nxl + nul
    stride = nxi + nui
    # 每根缆绳方向向量也要求保持单位长度。
    for i in 1:nq
        off = base + (i - 1) * stride
        d1 = x[off + 1]
        d2 = x[off + 2]
        d3 = x[off + 3]
        @constraint(model, d1^2 + d2^2 + d3^2 == 1.0)
    end

    # 每根缆绳张力在负载机体系中的表达，后面用于合力/合矩一致性。
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

    # 6 维 wrench 一致性：所有缆绳合起来的受力/力矩，要匹配负载控制量。
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

    # 下面是主要不等式约束：
    # - 负载避障
    # - 无人机避障
    # - 张力上下界
    # - 控制上下界
    # - 机间碰撞约束
    # - 绳索/负载几何约束
    # - 推力幅值上下界
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

        for kc in 1:num_dis
            for i in 1:nq
                for j in i+1:nq
                    expr_pair = pair_term_indexed(
                        x, base, stride, Rl,
                        ra[i, 1], ra[i, 2], ra[i, 3],
                        ra[j, 1], ra[j, 2], ra[j, 3],
                        cl0, rq, eps_margin, kc, num_dis, i, j,
                    )
                    @constraint(model, expr_pair <= 0.0)
                end
            end
        end

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

    params = (
        rho_lx = obj_params.rho_lx,
        rho_lu = obj_params.rho_lu,
        rho_ix = obj_params.rho_ix,
        rho_iu = obj_params.rho_iu,
        is_terminal = obj_params.is_terminal,
        xl_ideal = obj_params.xl_ideal,
        ul_ideal = obj_params.ul_ideal,
        y_xl = obj_params.y_xl,
        y_ul = obj_params.y_ul,
        xc_ideal = obj_params.xc_ideal,
        uc_ideal = obj_params.uc_ideal,
        y_xc = obj_params.y_xc,
        y_uc = obj_params.y_uc,
        Pt = Pt,
        pob1 = pob1,
        pob2 = pob2,
        ra = ra,
        ro = ro,
        rq = rq,
        cl0 = cl0,
        rl = rl,
        t_min = t_min,
        t_max = t_max,
        ui_bound = ui_bound,
        g = g,
        ml = ml,
        mq = mq,
        fmax = fmax,
        Jl = Jl,
        Jl_inv = Jl_inv,
    )
    return model, x, params
end

# 以下 helper 分成两类：
# 1. set_parameter_value / set_start_value 的轻量循环
# 2. 轻量 copy / compare，用来尽量少做无意义更新
#
# 这是热路径里的细活，存在的原因就是减少 JuMP 参数更新开销。
function _set_param_vec!(pref, values)
    @inbounds for i in eachindex(pref)
        set_parameter_value(pref[i], Float64(values[i]))
    end
end

function _set_param_mat!(pref, values)
    @inbounds for i in axes(pref, 1), j in axes(pref, 2)
        set_parameter_value(pref[i, j], Float64(values[i, j]))
    end
end

function _set_param_mat_nested!(pref, values)
    @inbounds for i in axes(pref, 1), j in axes(pref, 2)
        set_parameter_value(pref[i, j], Float64(values[i][j]))
    end
end

function _set_param_mat_nested_transposed!(pref, values)
    @inbounds for i in axes(pref, 1), j in axes(pref, 2)
        set_parameter_value(pref[i, j], Float64(values[j][i]))
    end
end

function _copy_vec_nested!(dest::AbstractVector{Float64}, src)
    @inbounds for i in eachindex(dest)
        dest[i] = Float64(src[i])
    end
    return dest
end

function _copy_mat_nested!(dest::AbstractMatrix{Float64}, src)
    @inbounds for i in axes(dest, 1), j in axes(dest, 2)
        dest[i, j] = Float64(src[i][j])
    end
    return dest
end

function _copy_mat_nested_transposed!(dest::AbstractMatrix{Float64}, src)
    @inbounds for i in axes(dest, 1), j in axes(dest, 2)
        dest[i, j] = Float64(src[j][i])
    end
    return dest
end

function _set_start_values!(xrefs, values)
    @inbounds for i in eachindex(xrefs)
        set_start_value(xrefs[i], Float64(values[i]))
    end
    return nothing
end

function _read_solution(xrefs)
    x_sol = Vector{Float64}(undef, length(xrefs))
    @inbounds for i in eachindex(xrefs)
        x_sol[i] = Float64(value(xrefs[i]))
    end
    return x_sol
end

function _store_warm_solution!(cached::CachedStepSolver, slot::Int, x_sol::Vector{Float64})
    cached.warm_slot_id = slot
    if cached.warm_solution === nothing || length(cached.warm_solution) != length(x_sol)
        cached.warm_solution = copy(x_sol)
    else
        copyto!(cached.warm_solution, x_sol)
    end
    return nothing
end

function _copy_warm_solution!(dest::CachedStepSolver, src::CachedStepSolver, slot::Int)
    if src.warm_solution !== nothing && src.warm_slot_id == slot
        dest.warm_slot_id = slot
        dest.warm_solution = copy(src.warm_solution)
    end
    return nothing
end

function _optimizer_iter_count(model)
    return try
        Int(round(Float64(JuMP.get_attribute(model, MOI.RawOptimizerAttribute("iter")))))
    catch
        try
            Int(round(Float64(JuMP.get_attribute(model, MOI.RawOptimizerAttribute("iter_count")))))
        catch
            -1
        end
    end
end

function _runtime_metrics(cached::CachedStepSolver, x_sol::Vector{Float64}, runtime_metric_mode::Symbol)
    if runtime_metric_mode == :none
        return nothing, nothing
    end
    eq_vec = Float64.(equality_residual(x_sol, cached.step.params, cached.step.dims))
    ineq_vec = Float64.(inequality_residual(x_sol, cached.step.params, cached.step.dims))
    return maximum(abs.(eq_vec)), maximum(max.(ineq_vec, 0.0))
end

function _vec_matches_nested(src_a, src_b)
    length(src_a) == length(src_b) || return false
    @inbounds for i in eachindex(src_a)
        if Float64(src_a[i]) != Float64(src_b[i])
            return false
        end
    end
    return true
end

function _mat_matches_nested(src_a, src_b)
    length(src_a) == length(src_b) || return false
    @inbounds for i in eachindex(src_a)
        row_a = src_a[i]
        row_b = src_b[i]
        length(row_a) == length(row_b) || return false
        for j in eachindex(row_a)
            if Float64(row_a[j]) != Float64(row_b[j])
                return false
            end
        end
    end
    return true
end

function _set_static_scalar_if_changed!(params, name::String, pref, value)
    value64 = Float64(value)
    if !haskey(params, name) || Float64(params[name]) != value64
        set_parameter_value(pref, value64)
        params[name] = value64
    end
    return nothing
end

function _set_static_vec_if_changed!(params, name::String, pref, src)
    if !haskey(params, name) || !_vec_matches_nested(params[name], src)
        _set_param_vec!(pref, src)
        params[name] = src
    end
    return nothing
end

function _set_static_mat_if_changed!(params, name::String, pref, src)
    if !haskey(params, name) || !_mat_matches_nested(params[name], src)
        _set_param_mat_nested!(pref, src)
        params[name] = src
    end
    return nothing
end

# 旧路径：每个 step 都是完整 JSON step 对象。
# 这条路径更直观，但运行时会有更多对象构造和字段拆装开销。
function update_parameterized_step!(cached::CachedStepSolver, step::SubP2StepModel.StepData)
    p = step.params
    refs = cached.params
    set_parameter_value(refs.rho_lx, Float64(p["rho_lx"]))
    set_parameter_value(refs.rho_lu, Float64(p["rho_lu"]))
    set_parameter_value(refs.rho_ix, Float64(p["rho_ix"]))
    set_parameter_value(refs.rho_iu, Float64(p["rho_iu"]))
    set_parameter_value(refs.is_terminal, Float64(p["is_terminal"]))
    _set_param_vec!(refs.xl_ideal, as_vec(p["xl_ideal"]))
    _set_param_vec!(refs.ul_ideal, as_vec(p["ul_ideal"]))
    _set_param_vec!(refs.y_xl, as_vec(p["y_xl"]))
    _set_param_vec!(refs.y_ul, as_vec(p["y_ul"]))
    _set_param_mat!(refs.xc_ideal, as_mat(p["xc_ideal"]))
    _set_param_mat!(refs.uc_ideal, as_mat(p["uc_ideal"]))
    _set_param_mat!(refs.y_xc, as_mat(p["y_xc"]))
    _set_param_mat!(refs.y_uc, as_mat(p["y_uc"]))
    _set_param_mat!(refs.Pt, as_mat(p["Pt"]))
    _set_param_vec!(refs.pob1, as_vec(p["pob1"]))
    _set_param_vec!(refs.pob2, as_vec(p["pob2"]))
    _set_param_mat!(refs.ra, transpose(as_mat(p["ra"])))
    set_parameter_value(refs.ro, Float64(p["ro"]))
    set_parameter_value(refs.rq, Float64(p["rq"]))
    set_parameter_value(refs.cl0, Float64(p["cl0"]))
    set_parameter_value(refs.rl, Float64(p["rl"]))
    set_parameter_value(refs.t_min, Float64(p["t_min"]))
    set_parameter_value(refs.t_max, Float64(p["t_max"]))
    set_parameter_value(refs.ui_bound, Float64(p["ui_bound"]))
    set_parameter_value(refs.g, Float64(p["g"]))
    set_parameter_value(refs.ml, Float64(p["ml"]))
    set_parameter_value(refs.mq, Float64(p["mq"]))
    set_parameter_value(refs.fmax, Float64(p["fmax"]))
    _set_param_mat!(refs.Jl, as_mat(p["Jl"]))
    _set_param_mat!(refs.Jl_inv, as_mat(p["Jl_inv"]))
    cached.step = step
    return nothing
end

# 主线路径：从 stacked runtime payload 中，按 idx 原地更新一个 cached step。
#
# 这里专门把"动态参数"和"静态参数"拆开：
# - 动态参数：每轮 ADMM / 每个 step 常变，直接刷新
# - 静态参数：多数时候不变，只有变化时才 set_parameter_value
#
# 这也是当前 Julia 热路径优化最核心的一层。
function update_parameterized_step_from_stacked!(cached::CachedStepSolver, data, idx::Int)
    params_batch = _getkey(data, "params_batch")
    refs = cached.params
    params = cached.step.params

    dynamic_scalar_names = (
        "rho_lx", "rho_lu", "rho_ix", "rho_iu", "is_terminal",
    )
    dynamic_scalar_refs = (
        refs.rho_lx, refs.rho_lu, refs.rho_ix, refs.rho_iu, refs.is_terminal,
    )
    @inbounds for (name, pref) in zip(dynamic_scalar_names, dynamic_scalar_refs)
        value = Float64(params_batch[name][idx])
        set_parameter_value(pref, value)
        params[name] = value
    end

    static_scalar_names = ("ro", "rq", "cl0", "rl", "t_min", "t_max", "ui_bound", "g", "ml", "mq", "fmax")
    static_scalar_refs = (refs.ro, refs.rq, refs.cl0, refs.rl, refs.t_min, refs.t_max, refs.ui_bound, refs.g, refs.ml, refs.mq, refs.fmax)
    @inbounds for (name, pref) in zip(static_scalar_names, static_scalar_refs)
        _set_static_scalar_if_changed!(params, name, pref, params_batch[name][idx])
    end

    x_init_src = _getkey(data, "x_init_batch")[idx]
    if !_vec_matches_nested(cached.step.x_init, x_init_src)
        _copy_vec_nested!(cached.step.x_init, x_init_src)
    end

    dynamic_vec_names = ("xl_ideal", "ul_ideal", "y_xl", "y_ul")
    dynamic_vec_refs = (refs.xl_ideal, refs.ul_ideal, refs.y_xl, refs.y_ul)
    @inbounds for (name, pref) in zip(dynamic_vec_names, dynamic_vec_refs)
        src = params_batch[name][idx]
        _set_param_vec!(pref, src)
        params[name] = src
    end

    static_vec_names = ("pob1", "pob2")
    static_vec_refs = (refs.pob1, refs.pob2)
    @inbounds for (name, pref) in zip(static_vec_names, static_vec_refs)
        _set_static_vec_if_changed!(params, name, pref, params_batch[name][idx])
    end

    dynamic_mat_names = ("xc_ideal", "uc_ideal", "y_xc", "y_uc")
    dynamic_mat_refs = (refs.xc_ideal, refs.uc_ideal, refs.y_xc, refs.y_uc)
    @inbounds for (name, pref) in zip(dynamic_mat_names, dynamic_mat_refs)
        src = params_batch[name][idx]
        _set_param_mat_nested!(pref, src)
        params[name] = src
    end

    static_mat_names = ("Pt", "Jl", "Jl_inv")
    static_mat_refs = (refs.Pt, refs.Jl, refs.Jl_inv)
    @inbounds for (name, pref) in zip(static_mat_names, static_mat_refs)
        _set_static_mat_if_changed!(params, name, pref, params_batch[name][idx])
    end

    ra_src = params_batch["ra"][idx]
    if !haskey(params, "ra") || !_mat_matches_nested(params["ra"], ra_src)
        _set_param_mat_nested_transposed!(refs.ra, ra_src)
        params["ra"] = ra_src
    end

    meta = cached.step.meta
    target_steps = _getkey(data, "target_steps")
    meta["target_step"] = Int(target_steps[idx])
    return nothing
end

# 从一个 step data 构造"可长期复用"的 solver 槽位。
# 注意这里做的是：建模型、绑定参数引用、配 solver 选项、首次写入参数，
# 而不是立刻求解。
function build_cached_step_solver(
    step::SubP2StepModel.StepData;
    include_simple_ineq::Bool = true,
    print_level = MadNLP.ERROR,
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
)
    model, _, params = build_jump_model_eq_parameterized(step.dims; include_simple_ineq = include_simple_ineq)
    xrefs = model[:x]
    set_optimizer_attribute(model, "print_level", Int(print_level))
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "acceptable_tol", acceptable_tol)
    set_optimizer_attribute(model, "acceptable_iter", acceptable_iter)
    set_optimizer_attribute(model, "tol", tol)
    cached = CachedStepSolver(
        step,
        model,
        xrefs,
        params,
        _backend_key(),
        warm_slot(step),
        nothing,
        max_iter,
        Float64(acceptable_tol),
        acceptable_iter,
        Float64(tol),
    )
    update_parameterized_step!(cached, step)
    return cached
end

# solver 选项也走"只在变化时更新"，避免每轮反复 set optimizer attribute。
function set_solver_options!(
    cached::CachedStepSolver;
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
)
    if cached.last_max_iter != max_iter
        set_optimizer_attribute(cached.model, "max_iter", max_iter)
        cached.last_max_iter = max_iter
    end
    if cached.last_acceptable_tol != Float64(acceptable_tol)
        set_optimizer_attribute(cached.model, "acceptable_tol", acceptable_tol)
        cached.last_acceptable_tol = Float64(acceptable_tol)
    end
    if cached.last_acceptable_iter != acceptable_iter
        set_optimizer_attribute(cached.model, "acceptable_iter", acceptable_iter)
        cached.last_acceptable_iter = acceptable_iter
    end
    if cached.last_tol != Float64(tol)
        set_optimizer_attribute(cached.model, "tol", tol)
        cached.last_tol = Float64(tol)
    end
    return nothing
end

# warm slot 本质上就是"这个解属于哪个 time step 的 warm start"。
# 当前实现不是多槽 stage cache，而是单槽缓存。
function warm_slot(step::SubP2StepModel.StepData)
    meta = step.meta
    if meta isa AbstractDict
        if haskey(meta, "target_step")
            return Int(meta["target_step"])
        elseif haskey(meta, :target_step)
            return Int(meta[:target_step])
        end
    end
    return 0
end

# 单步求解的完整返回版本。
# 适合 legacy 路径或者调试时保留更多字段。
function solve_cached_step!(
    cached::CachedStepSolver;
    slot::Int = 0,
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
    runtime_only::Bool = false,
    runtime_metric_mode::Symbol = :full,
)
    set_solver_options!(
        cached;
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    warm_sol = cached.warm_slot_id == slot ? cached.warm_solution : nothing
    if warm_sol !== nothing
        _set_start_values!(cached.xrefs, warm_sol)
    else
        _set_start_values!(cached.xrefs, cached.step.x_init)
    end
    optimize!(cached.model)
    x_sol = _read_solution(cached.xrefs)
    _store_warm_solution!(cached, slot, x_sol)
    metrics = if runtime_only
        if runtime_metric_mode == :none
            (
                eq_inf = nothing,
                ineq_vio = nothing,
            )
        else
            eq_vec = Float64.(equality_residual(x_sol, cached.step.params, cached.step.dims))
            ineq_vec = Float64.(inequality_residual(x_sol, cached.step.params, cached.step.dims))
            (
                eq_inf = maximum(abs.(eq_vec)),
                ineq_vio = maximum(max.(ineq_vec, 0.0)),
            )
        end
    else
        summarize_solution(cached.step, x_sol)
    end
    status = string(termination_status(cached.model))
    primal = string(primal_status(cached.model))
    iter = _optimizer_iter_count(cached.model)
    result = Dict(
        "status" => status,
        "primal_status" => primal,
        "iter" => iter,
        "eq_inf" => metrics.eq_inf,
        "ineq_vio" => metrics.ineq_vio,
        "include_simple_ineq" => true,
        "x_sol" => x_sol,
    )
    if !runtime_only
        result["objective"] = metrics.objective
        result["orig_rmse"] = metrics.orig_rmse
        result["orig_max_abs"] = metrics.orig_max_abs
    end
    return result
end

# 单步求解的紧凑返回版本。
# 当前 Python 主线更偏向用这条，因为回包更小、重建更轻。
function solve_cached_step_compact!(
    cached::CachedStepSolver;
    slot::Int = 0,
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
    runtime_metric_mode::Symbol = :full,
)
    set_solver_options!(
        cached;
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    warm_sol = cached.warm_slot_id == slot ? cached.warm_solution : nothing
    if warm_sol !== nothing
        _set_start_values!(cached.xrefs, warm_sol)
    else
        _set_start_values!(cached.xrefs, cached.step.x_init)
    end
    optimize!(cached.model)
    x_sol = _read_solution(cached.xrefs)
    _store_warm_solution!(cached, slot, x_sol)
    eq_inf, ineq_vio = _runtime_metrics(cached, x_sol, runtime_metric_mode)
    iter = _optimizer_iter_count(cached.model)
    return (x_sol = x_sol, iter = iter, eq_inf = eq_inf, ineq_vio = ineq_vio)
end

function _solver_cache_key(step::SubP2StepModel.StepData)
    return (warm_slot(step), string(structure_cache_key(step), "|", _backend_key()))
end

function _solver_cache_key(slot::Int, dims::NTuple{6, Int})
    return (slot, string(structure_cache_key(dims), "|", _backend_key()))
end

# 如果某个 cached solver 因内部状态问题求解失败，
# 这里会重建同结构 solver 再重试一次，尽量不让整个 batch 因单个槽位脏掉而崩盘。
function _solve_with_rebuild_fallback!(
    cached::CachedStepSolver,
    step::SubP2StepModel.StepData,
    cache_key;
    slot::Int,
    max_iter,
    acceptable_tol,
    acceptable_iter,
    tol,
    runtime_only::Bool = false,
    runtime_metric_mode::Symbol = :full,
)
    try
        cached.step = step
        return solve_cached_step!(
            cached;
            slot = slot,
            max_iter = max_iter,
            acceptable_tol = acceptable_tol,
            acceptable_iter = acceptable_iter,
            tol = tol,
            runtime_only = runtime_only,
            runtime_metric_mode = runtime_metric_mode,
        )
    catch err
        if !(err isa Exception)
            rethrow(err)
        end
        rebuilt = build_cached_step_solver(
            step;
            print_level = MadNLP.ERROR,
            include_simple_ineq = true,
            max_iter = max_iter,
            acceptable_tol = acceptable_tol,
            acceptable_iter = acceptable_iter,
            tol = tol,
        )
        _copy_warm_solution!(rebuilt, cached, slot)
        cached = lock(_STEP_SOLVER_CACHE_LOCK) do
            _STEP_SOLVER_CACHE[cache_key] = rebuilt
            rebuilt
        end
        cached.step = step
        result = solve_cached_step!(
            cached;
            slot = slot,
            max_iter = max_iter,
            acceptable_tol = acceptable_tol,
            acceptable_iter = acceptable_iter,
            tol = tol,
            runtime_only = runtime_only,
            runtime_metric_mode = runtime_metric_mode,
        )
        result["rebuilt_after_failure"] = true
        return result
    end
end

# stacked_runtime_v1 是当前 Python/JAX -> Julia 主线使用的 payload 协议。
# 它不是 101 个 step 的 Dict 列表，而是把所有 step 的 x_init / params 按 batch 组织起来。
function _has_stacked_runtime_format(data)
    fmt = if haskey(data, "format")
        String(data["format"])
    elseif haskey(data, :format)
        String(data[:format])
    else
        ""
    end
    return fmt == "stacked_runtime_v1"
end

function _getkey(data, key::String)
    return haskey(data, key) ? data[key] : data[Symbol(key)]
end

function _slice_batch_value(v, idx::Int)
    return v[idx]
end

function _metric_or_jsonable(value)
    return value === nothing ? nothing : Float64(value)
end

# 把 stacked payload 的第 idx 个切片重新包装成一个 StepData。
# 这个函数主要用于"首次建模板"阶段，而不是每轮热路径都频繁新建对象。
function step_data_from_stacked_payload(data, idx::Int)
    dims = Tuple(Int(v) for v in _getkey(data, "dims"))
    meta0 = _getkey(data, "meta")
    meta = Dict{String, Any}()
    for (k, v) in pairs(meta0)
        meta[string(k)] = v
    end
    target_steps = _getkey(data, "target_steps")
    meta["target_step"] = Int(target_steps[idx])
    params_batch = _getkey(data, "params_batch")
    params_t = Dict{String, Any}()
    for (k, v) in pairs(params_batch)
        params_t[string(k)] = _slice_batch_value(v, idx)
    end
    x_init = as_vec(_getkey(data, "x_init_batch")[idx])
    eq_dim = 1 + dims[5] + 6
    ineq_dim = 2 + 2 * dims[5] + 2 * dims[6] * (dims[5] * (dims[5] - 1) ÷ 2) + 2 * dims[5] + 2 * dims[5] + 2 * dims[5] * dims[4] + 2 * dims[5]
    return SubP2StepModel.StepData(
        meta,
        dims,
        params_t,
        x_init,
        copy(x_init),
        0.0,
        zeros(Float64, eq_dim),
        zeros(Float64, ineq_dim),
        0.0,
        zeros(Float64, eq_dim),
        zeros(Float64, ineq_dim),
    )
end

# 为某个 (nsteps, dims, backend) 组合准备一整批 template。
# 后续每轮 stacked solve 直接复用这批模板，只改参数和 warm start。
function _ensure_stacked_batch_template!(
    data,
    dims::NTuple{6, Int},
    target_steps,
    nsteps::Int;
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
)
    template_key = (nsteps, dims, _backend_key())
    template = lock(_STACKED_BATCH_TEMPLATE_CACHE_LOCK) do
        get(_STACKED_BATCH_TEMPLATE_CACHE, template_key, nothing)
    end
    if template !== nothing
        return template
    end

    built = Vector{CachedStepSolver}(undef, nsteps)
    for idx in 1:nsteps
        step = step_data_from_stacked_payload(data, idx)
        built[idx] = build_cached_step_solver(
            step;
            print_level = MadNLP.ERROR,
            include_simple_ineq = true,
            max_iter = max_iter,
            acceptable_tol = acceptable_tol,
            acceptable_iter = acceptable_iter,
            tol = tol,
        )
    end

    return lock(_STACKED_BATCH_TEMPLATE_CACHE_LOCK) do
        get!(_STACKED_BATCH_TEMPLATE_CACHE, template_key, built)
    end
end

# prepare_batch_payload! 的作用不是求解，而是把模型骨架提前预热好。
# 这样 Python 可以把"首次建模板"成本和真正 solve 成本拆开看。
function prepare_batch_payload!(
    data;
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
)
    if !_has_stacked_runtime_format(data)
        # ---------------------------
        # 可先跳过：legacy path
        # ---------------------------
        # 这条路径处理的是“每个 step 都是独立 JSON 对象”的旧格式。
        # 当前 Python/JAX 正式主线默认不会走这里。
        steps = data["steps"]
        count = length(steps)
        t0 = time_ns()
        built = 0
        for step_json in steps
            step = SubP2StepModel.step_data_from_json(step_json)
            cache_key = _solver_cache_key(step)
            cached = lock(_STEP_SOLVER_CACHE_LOCK) do
                get(_STEP_SOLVER_CACHE, cache_key, nothing)
            end
            if cached === nothing
                new_cached = build_cached_step_solver(
                    step;
                    print_level = MadNLP.ERROR,
                    include_simple_ineq = true,
                    max_iter = max_iter,
                    acceptable_tol = acceptable_tol,
                    acceptable_iter = acceptable_iter,
                    tol = tol,
                )
                lock(_STEP_SOLVER_CACHE_LOCK) do
                    get!(_STEP_SOLVER_CACHE, cache_key, new_cached)
                end
                built += 1
            end
        end
        return Dict(
            "count" => count,
            "built" => built,
            "prepare_ms" => (time_ns() - t0) / 1e6,
            "mode" => "legacy_exact_cache",
        )
    end

    # ---------------------------
    # 当前正式主线从这里开始：
    # - 读取 stacked batch
    # - 复用 template
    # - 按 step 并行 solve
    # - 返回紧凑 batch 结果
    # ---------------------------
    # 当前主线的必经调用链可以直接按这个顺序追：
    # solve_batch_payload (stacked path)
    #   -> _ensure_stacked_batch_template!
    #   -> solve_one_stacked(idx)
    #   -> _solve_stacked_cached_step!
    #   -> update_parameterized_step_from_stacked!
    #   -> solve_cached_step_compact!
    x_init_batch = _getkey(data, "x_init_batch")
    nsteps = length(x_init_batch)
    dims = Tuple(Int(v) for v in _getkey(data, "dims"))
    target_steps = _getkey(data, "target_steps")
    t0 = time_ns()
    _ensure_stacked_batch_template!(
        data,
        dims,
        target_steps,
        nsteps;
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    return Dict(
        "count" => nsteps,
        "built" => nsteps,
        "prepare_ms" => (time_ns() - t0) / 1e6,
        "mode" => "stacked_runtime_template",
    )
end

# 用一个现成 template[idx] 去解当前 batch 的第 idx 个 step。
# 主线路径里真正的热工作就是：
#   update_parameterized_step_from_stacked! -> solve_cached_step_compact!
function _solve_stacked_cached_step!(
    template::Vector{CachedStepSolver},
    idx::Int,
    data,
    dims::NTuple{6, Int},
    slot::Int;
    max_iter,
    acceptable_tol,
    acceptable_iter,
    tol,
    runtime_only::Bool = true,
    runtime_metric_mode::Symbol = :full,
    compact_result::Bool = false,
)
    cached = template[idx]
    update_parameterized_step_from_stacked!(cached, data, idx)
    try
        if compact_result
            return solve_cached_step_compact!(
                cached;
                slot = slot,
                max_iter = max_iter,
                acceptable_tol = acceptable_tol,
                acceptable_iter = acceptable_iter,
                tol = tol,
                runtime_metric_mode = runtime_metric_mode,
            )
        end
        return solve_cached_step!(
            cached;
            slot = slot,
            max_iter = max_iter,
            acceptable_tol = acceptable_tol,
            acceptable_iter = acceptable_iter,
            tol = tol,
            runtime_only = runtime_only,
            runtime_metric_mode = runtime_metric_mode,
        )
    catch err
        if !(err isa Exception)
            rethrow(err)
        end
        rebuilt = build_cached_step_solver(
            cached.step;
            print_level = MadNLP.ERROR,
            include_simple_ineq = true,
            max_iter = max_iter,
            acceptable_tol = acceptable_tol,
            acceptable_iter = acceptable_iter,
            tol = tol,
        )
        _copy_warm_solution!(rebuilt, cached, slot)
        template[idx] = rebuilt
        result = if compact_result
            solve_cached_step_compact!(
                rebuilt;
                slot = slot,
                max_iter = max_iter,
                acceptable_tol = acceptable_tol,
                acceptable_iter = acceptable_iter,
                tol = tol,
                runtime_metric_mode = runtime_metric_mode,
            )
        else
            solve_cached_step!(
                rebuilt;
                slot = slot,
                max_iter = max_iter,
                acceptable_tol = acceptable_tol,
                acceptable_iter = acceptable_iter,
                tol = tol,
                runtime_only = runtime_only,
                runtime_metric_mode = runtime_metric_mode,
            )
        end
        if compact_result
            return (
                x_sol = result.x_sol,
                iter = result.iter,
                eq_inf = result.eq_inf,
                ineq_vio = result.ineq_vio,
                rebuilt_after_failure = true,
            )
        end
        result["rebuilt_after_failure"] = true
        return result
    end
end

# 这是 Julia 侧 batch 求解的总入口。
#
# 文件里同时保留了两条路径：
# 1. legacy path: data["steps"] = 每个 step 一个独立 JSON 对象
# 2. stacked path: 当前主线，批量 x_init / params，紧凑回包
#
# 真正日常主线看 stacked path 就够了。
function solve_batch_payload(
    data;
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
    runtime_metric_mode::Symbol = :full,
    result_format::Symbol = :default,
)
    if !_has_stacked_runtime_format(data)
        # ---------------------------
        # 可先跳过：legacy path
        # ---------------------------
        # 这条路径处理的是“每个 step 都是独立 JSON 对象”的旧格式。
        # 当前 Python/JAX 正式主线默认不会走这里。
        steps = data["steps"]
        results = Vector{Any}(undef, length(steps))
        t0 = time_ns()
        build_ms = zeros(Float64, length(steps))
        solve_ms = zeros(Float64, length(steps))
        solve_one_step = function (idx)
            step_json = steps[idx]
            step = SubP2StepModel.step_data_from_json(step_json)
            slot = warm_slot(step)
            cache_key = _solver_cache_key(step)
            cached = lock(_STEP_SOLVER_CACHE_LOCK) do
                get(_STEP_SOLVER_CACHE, cache_key, nothing)
            end
            if cached === nothing
                t_build = time_ns()
                new_cached = build_cached_step_solver(
                    step;
                    print_level = MadNLP.ERROR,
                    include_simple_ineq = true,
                    max_iter = max_iter,
                    acceptable_tol = acceptable_tol,
                    acceptable_iter = acceptable_iter,
                    tol = tol,
                )
                cached = lock(_STEP_SOLVER_CACHE_LOCK) do
                    get!(_STEP_SOLVER_CACHE, cache_key, new_cached)
                end
                build_ms[idx] = (time_ns() - t_build) / 1e6
            else
                update_parameterized_step!(cached, step)
            end
            t_solve = time_ns()
            result = _solve_with_rebuild_fallback!(
                cached,
                step,
                cache_key;
                slot = slot,
                max_iter = max_iter,
                acceptable_tol = acceptable_tol,
                acceptable_iter = acceptable_iter,
                tol = tol,
                runtime_only = false,
                runtime_metric_mode = runtime_metric_mode,
            )
            result["build_ms"] = build_ms[idx]
            result["solve_ms"] = (time_ns() - t_solve) / 1e6
            result["wall_ms"] = result["build_ms"] + result["solve_ms"]
            solve_ms[idx] = Float64(result["solve_ms"])
            results[idx] = result
            return nothing
        end
        # 这里仍然是按 step 并行；GPU 某些模式下才会退回外层串行。
        if _gpu_linear_solver_enabled() && _gpu_outer_mode() == "serial"
            for idx in eachindex(steps)
                solve_one_step(idx)
            end
        else
            _run_threaded_range!(solve_one_step, length(steps))
        end
        total_ms = (time_ns() - t0) / 1e6
        return Dict(
            "count" => length(steps),
            "build_ms_total" => sum(build_ms),
            "solve_ms_total" => sum(solve_ms),
            "total_ms" => total_ms,
            "results" => results,
        )
    end

    # ---------------------------
    # 当前正式主线从这里开始：
    # - 读取 stacked batch
    # - 复用 template
    # - 按 step 并行 solve
    # - 返回紧凑 batch 结果
    # ---------------------------
    # 当前主线的必经调用链可以直接按这个顺序追：
    # solve_batch_payload (stacked path)
    #   -> _ensure_stacked_batch_template!
    #   -> solve_one_stacked(idx)
    #   -> _solve_stacked_cached_step!
    #   -> update_parameterized_step_from_stacked!
    #   -> solve_cached_step_compact!
    x_init_batch = _getkey(data, "x_init_batch")
    nsteps = length(x_init_batch)
    dims = Tuple(Int(v) for v in _getkey(data, "dims"))
    target_steps = _getkey(data, "target_steps")
    template = _ensure_stacked_batch_template!(
        data,
        dims,
        target_steps,
        nsteps;
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    compact_result = result_format == :stacked_runtime_result_v1
    results = compact_result ? nothing : Vector{Any}(undef, nsteps)
    xdim = length(template[1].step.x_init)
    x_sol_batch = compact_result ? Matrix{Float64}(undef, nsteps, xdim) : nothing
    iter_batch = compact_result ? Vector{Int}(undef, nsteps) : nothing
    eq_inf_batch = compact_result ? Vector{Any}(undef, nsteps) : nothing
    ineq_vio_batch = compact_result ? Vector{Any}(undef, nsteps) : nothing
    t0 = time_ns()
    build_ms = zeros(Float64, nsteps)
    solve_ms = zeros(Float64, nsteps)
    solve_one_stacked = function (idx)
        slot = Int(target_steps[idx])
        t_solve = time_ns()
        result = _solve_stacked_cached_step!(
            template,
            idx,
            data,
            dims,
            slot;
            max_iter = max_iter,
            acceptable_tol = acceptable_tol,
            acceptable_iter = acceptable_iter,
            tol = tol,
            runtime_only = true,
            runtime_metric_mode = runtime_metric_mode,
            compact_result = compact_result,
        )
        if compact_result
            solve_ms[idx] = (time_ns() - t_solve) / 1e6
            x_sol_batch[idx, :] .= result.x_sol
            iter_batch[idx] = Int(result.iter)
            eq_inf_batch[idx] = _metric_or_jsonable(result.eq_inf)
            ineq_vio_batch[idx] = _metric_or_jsonable(result.ineq_vio)
        else
            result["build_ms"] = build_ms[idx]
            result["solve_ms"] = (time_ns() - t_solve) / 1e6
            result["wall_ms"] = result["build_ms"] + result["solve_ms"]
            solve_ms[idx] = Float64(result["solve_ms"])
            results[idx] = result
        end
        return nothing
    end
    # CPU 主线默认在这里做 step 级并行。
    # 只有少数 GPU 线性求解器模式才会强制退回外层串行。
    if _gpu_linear_solver_enabled() && _gpu_outer_mode() == "serial"
        for idx in 1:nsteps
            solve_one_stacked(idx)
        end
    else
        _run_threaded_range!(solve_one_stacked, nsteps)
    end
    total_ms = (time_ns() - t0) / 1e6
    out = Dict{String, Any}(
        "count" => nsteps,
        "build_ms_total" => sum(build_ms),
        "solve_ms_total" => sum(solve_ms),
        "total_ms" => total_ms,
    )
    if compact_result
        out["result_format"] = String(result_format)
        out["x_sol_batch"] = x_sol_batch
        out["iter_batch"] = iter_batch
        out["eq_inf_batch"] = eq_inf_batch
        out["ineq_vio_batch"] = ineq_vio_batch
    else
        out["results"] = results
    end
    return out
end

# CLI / 文件路径兼容入口。
# persistent worker 平时更常走 solve_batch_payload(data)，
# 但这里保留直接读 JSON 文件求解的能力，便于离线排查。
function solve_batch(
    path::String;
    max_iter = 200,
    acceptable_tol = 1e-4,
    acceptable_iter = 5,
    tol = 1e-8,
    runtime_metric_mode::Symbol = :full,
    result_format::Symbol = :default,
)
    data = JSON3.read(read(path, String))
    return solve_batch_payload(
        data;
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
        runtime_metric_mode = runtime_metric_mode,
        result_format = result_format,
    )
end

# 这个 main 主要给"直接命令行跑单个 batch.json"使用。
# persistent worker 模式并不会从这里进，而是 include 本文件后直接调用 solve_batch_payload。
function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/solve_batch_madnlp_jump_native.jl <batch.json> [out.json] [--max-iter=N] [--acceptable-tol=X] [--acceptable-iter=N] [--tol=X]")
    end
    out_path = nothing
    max_iter = 200
    acceptable_tol = 1e-4
    acceptable_iter = 5
    tol = 1e-8
    for arg in ARGS[2:end]
        if startswith(arg, "--max-iter=")
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
    result = solve_batch(
        ARGS[1];
        max_iter = max_iter,
        acceptable_tol = acceptable_tol,
        acceptable_iter = acceptable_iter,
        tol = tol,
    )
    println("MadNLP JuMP native batch count: ", result["count"])
    println("batch_build_ms_total: ", round(result["build_ms_total"]; digits = 3))
    println("batch_solve_ms_total: ", round(result["solve_ms_total"]; digits = 3))
    println("batch_total_ms: ", round(result["total_ms"]; digits = 3))
    if out_path !== nothing
        open(out_path, "w") do io
            JSON3.pretty(io, result)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
