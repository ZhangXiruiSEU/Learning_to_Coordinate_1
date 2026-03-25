module SubP2StepModel

using JSON3
using LinearAlgebra
using ExaModels

# -----------------------------------------------------------------------------
# 这个文件是 SubP2 的"底层数学定义层"。
#
# 它和上层文件的关系是：
# - solve_batch_madnlp_jump_native.jl 会直接复用这里的一批 helper、残差函数和 StepData
# - solve_step_madnlp_jump_native_eq.jl 也会复用这里的数据结构与数学定义
# - 只有 ExaModels 相关那部分，属于更老的/兼容的建模路径，当前 persistent Julia 主线不直接走
#
# 如果你现在只关心"当前正式 batch 主线"，最小阅读顺序建议是：
# 1. StepData                         # 单步数据长什么样
# 2. step_data_from_json(...)         # Python/JSON snapshot 如何落成 Julia step
# 3. pair_term_indexed / gio_terms_indexed / thrust_terms_indexed
# 4. objective(...)
# 5. equality_residual(...)
# 6. inequality_residual(...)
#
# 当前主线里可先跳过的部分：
# - pair_term_from_x / gio_terms_from_x / thrust_terms_from_x
#   主要给 ExaModels/按 accessor 取值的旧路径用
# - control_bound_from_x(...)
#   小型 helper，当前主线没直接走到
# - add_native_equality_constraints!(...)
# - add_native_inequality_constraints!(...)
# - build_examodel(...)
#   这三块属于 ExaModels 建模路径，当前 persistent Julia + JuMP/MadNLP 主线不直接调用
# - inequality_resual_component(...)
#   兼容旧拼写的别名，不是新的核心入口
# -----------------------------------------------------------------------------

export StepData,
    as_vec,
    as_mat,
    rowmajor_vec,
    unpack_w,
    pair_term_from_x,
    gio_terms_from_x,
    control_bound_from_x,
    thrust_terms_from_x,
    pair_term_indexed,
    gio_terms_indexed,
    thrust_terms_indexed,
    objective,
    equality_residual,
    inequality_residual,
    inequality_resual_component,
    equality_component,
    inequality_component,
    load_step_data,
    step_data_from_json,
    build_examodel

# 单个 SubP2 step 的完整静态快照。
# 当前 batch 主线里，每个 time step 最终都会先被整理成这个结构。
struct StepData
    meta
    dims::NTuple{6, Int}
    params
    x_init::Vector{Float64}
    x_orig::Vector{Float64}
    init_obj_ref::Float64
    init_eq_ref::Vector{Float64}
    init_ineq_ref::Vector{Float64}
    orig_obj_ref::Float64
    orig_eq_ref::Vector{Float64}
    orig_ineq_ref::Vector{Float64}
end

# 把 JSON 数组等通用容器稳定转换成 Vector{Float64}。
# 当前主线会频繁走这里，属于基础清洗 helper。
function as_vec(x)
    return [Float64(v) for v in x]
end

# 把嵌套数组转换成 Matrix{Float64}，并显式检查每行列数一致。
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

# 按行展平矩阵。当前 persistent batch 主线基本不依赖它，
# 更偏工具/兼容 helper，可先跳过。
function rowmajor_vec(mat)
    out = Any[]
    sizehint!(out, length(mat))
    for i in axes(mat, 1)
        for j in axes(mat, 2)
            push!(out, mat[i, j])
        end
    end
    return out
end

# 四元数 -> 旋转矩阵。
# equality/inequality residual 和若干单步建模函数都会复用。
function q_to_rotation(q)
    q0, q1, q2, q3 = q
    return [
        2 * (q0^2 + q1^2) - 1  2 * (q1 * q2 - q0 * q3)  2 * (q1 * q3 + q0 * q2);
        2 * (q1 * q2 + q0 * q3)  2 * (q0^2 + q2^2) - 1  2 * (q2 * q3 - q0 * q1);
        2 * (q1 * q3 - q0 * q2)  2 * (q2 * q3 + q0 * q1)  2 * (q0^2 + q3^2) - 1;
    ]
end

# 把打平后的决策变量 w 拆成 xl / ul / xc / uc 四块。
# 当前 residual/objective 路径会直接依赖这个函数。
function unpack_w(w, dims)
    nxl, nul, nxi, nui, nq, _ = dims
    xl = Any[w[i] for i in 1:nxl]
    ul = Any[w[nxl + i] for i in 1:nul]
    base = nxl + nul
    xc = Array{Any}(undef, nq, nxi)
    uc = Array{Any}(undef, nq, nui)
    for i in 1:nq
        off = base + (i - 1) * (nxi + nui)
        for j in 1:nxi
            xc[i, j] = w[off + j]
        end
        for j in 1:nui
            uc[i, j] = w[off + nxi + j]
        end
    end
    return xl, ul, xc, uc
end

# 下面这组 accessor 主要给"按索引直接拼表达式"的底层 helper 用。
# 其中 indexed 版本在当前 batch 主线会间接用到；from_x 版本主要服务 ExaModels 路径。
@inline x_xl(w, idx) = w[idx]
@inline x_ul(w, nxl, idx) = w[nxl + idx]
@inline x_xi(w, dims, i, j) = begin
    nxl, nul, nxi, nui, _, _ = dims
    base = nxl + nul + (i - 1) * (nxi + nui)
    w[base + j]
end
@inline x_ui(w, dims, i, j) = begin
    nxl, nul, nxi, nui, _, _ = dims
    base = nxl + nul + (i - 1) * (nxi + nui)
    w[base + nxi + j]
end

# 从打平变量向量里直接取负载四元数并展开旋转矩阵。
# 主要给 ExaModels 约束构造路径用，当前 batch 主线不直接走这里。
function q_to_rotation_from_x(w, dims)
    q0 = x_xl(w, 7)
    q1 = x_xl(w, 8)
    q2 = x_xl(w, 9)
    q3 = x_xl(w, 10)
    return [
        2 * (q0^2 + q1^2) - 1  2 * (q1 * q2 - q0 * q3)  2 * (q1 * q3 + q0 * q2);
        2 * (q1 * q2 + q0 * q3)  2 * (q0^2 + q2^2) - 1  2 * (q2 * q3 - q0 * q1);
        2 * (q1 * q3 - q0 * q2)  2 * (q2 * q3 + q0 * q1)  2 * (q0^2 + q3^2) - 1;
    ]
end

# 手写 3D 叉乘，避免在符号/建模表达式里频繁构小向量。
function cross3(ax, ay, az, bx, by, bz)
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )
end

# 旧的 from_x 版本推力项 helper：
# - 主要服务 ExaModels / accessor 风格约束构造
# - 当前 persistent batch 主线更直接用下面的 thrust_terms_indexed(...)
# 所以这里属于"可先跳过，但不是废代码"。
Base.@noinline function thrust_terms_from_x(w, dims, Rl, wl, awl, ra1, ra2, ra3, cl0, mq, ml, g, ul1, ul2, ul3, i)
    d1 = x_xi(w, dims, i, 1)
    d2 = x_xi(w, dims, i, 2)
    d3 = x_xi(w, dims, i, 3)
    w1 = x_xi(w, dims, i, 4)
    w2 = x_xi(w, dims, i, 5)
    w3 = x_xi(w, dims, i, 6)
    dw1 = x_xi(w, dims, i, 7)
    dw2 = x_xi(w, dims, i, 8)
    dw3 = x_xi(w, dims, i, 9)
    t = x_xi(w, dims, i, 13)

    cwr1, cwr2, cwr3 = cross3(wl[1], wl[2], wl[3], ra1, ra2, ra3)
    cwcr1, cwcr2, cwcr3 = cross3(wl[1], wl[2], wl[3], cwr1, cwr2, cwr3)
    car1, car2, car3 = cross3(awl[1], awl[2], awl[3], ra1, ra2, ra3)
    rotb1 = cwcr1 + car1
    rotb2 = cwcr2 + car2
    rotb3 = cwcr3 + car3
    rot1 = Rl[1, 1] * rotb1 + Rl[1, 2] * rotb2 + Rl[1, 3] * rotb3
    rot2 = Rl[2, 1] * rotb1 + Rl[2, 2] * rotb2 + Rl[2, 3] * rotb3
    rot3 = Rl[3, 1] * rotb1 + Rl[3, 2] * rotb2 + Rl[3, 3] * rotb3

    cdd1, cdd2, cdd3 = cross3(dw1, dw2, dw3, d1, d2, d3)
    cwd1, cwd2, cwd3 = cross3(w1, w2, w3, d1, d2, d3)
    cwwd1, cwwd2, cwwd3 = cross3(w1, w2, w3, cwd1, cwd2, cwd3)
    cable1 = cl0 * (cdd1 + cwwd1)
    cable2 = cl0 * (cdd2 + cwwd2)
    cable3 = cl0 * (cdd3 + cwwd3)

    f1 = mq * (ul1 / ml + rot1 + cable1) + d1 * t
    f2 = mq * (ul2 / ml + rot2 + cable2) + d2 * t
    # Match inequality_residual: al3 = ul3 / ml - g, then + g inside fi[3].
    # Net effect is ul3 / ml + rot3 + cable3 without an extra gravity term.
    f3 = mq * (ul3 / ml + rot3 + cable3) + d3 * t
    return f1, f2, f3
end

# 旧的 from_x 版本绳索-绳索间距项 helper，主要给 ExaModels 路径用。
Base.@noinline function pair_term_from_x(w, dims, Rl, ra_i1, ra_i2, ra_i3, ra_j1, ra_j2, ra_j3, cl0, rq, eps_margin, kc, num_dis, i, j)
    frac = kc / num_dis
    pib_i1 = ra_i1 + frac * cl0 * (Rl[1, 1] * x_xi(w, dims, i, 1) + Rl[2, 1] * x_xi(w, dims, i, 2) + Rl[3, 1] * x_xi(w, dims, i, 3))
    pib_i2 = ra_i2 + frac * cl0 * (Rl[1, 2] * x_xi(w, dims, i, 1) + Rl[2, 2] * x_xi(w, dims, i, 2) + Rl[3, 2] * x_xi(w, dims, i, 3))
    pib_j1 = ra_j1 + frac * cl0 * (Rl[1, 1] * x_xi(w, dims, j, 1) + Rl[2, 1] * x_xi(w, dims, j, 2) + Rl[3, 1] * x_xi(w, dims, j, 3))
    pib_j2 = ra_j2 + frac * cl0 * (Rl[1, 2] * x_xi(w, dims, j, 1) + Rl[2, 2] * x_xi(w, dims, j, 2) + Rl[3, 2] * x_xi(w, dims, j, 3))
    return ((kc / num_dis) * 4.0 * rq)^2 + eps_margin - ((pib_i1 - pib_j1)^2 + (pib_i2 - pib_j2)^2)
end

# 旧的 from_x 版本几何约束 helper，主要给 ExaModels 路径用。
Base.@noinline function gio_terms_from_x(w, dims, Rl, ra1, ra2, ra3, cl0, rl, num_dis, i)
    frac = num_dis / num_dis
    ra_norm = sqrt(ra1^2 + ra2^2 + ra3^2) + 1e-9
    pib1 = ra1 + frac * cl0 * (Rl[1, 1] * x_xi(w, dims, i, 1) + Rl[2, 1] * x_xi(w, dims, i, 2) + Rl[3, 1] * x_xi(w, dims, i, 3))
    pib2 = ra2 + frac * cl0 * (Rl[1, 2] * x_xi(w, dims, i, 1) + Rl[2, 2] * x_xi(w, dims, i, 2) + Rl[3, 2] * x_xi(w, dims, i, 3))
    ei_pib = (ra1 / ra_norm) * pib1 + (ra2 / ra_norm) * pib2
    return ei_pib - (rl + cl0), -rl - ei_pib
end

# 控制量上下界的小 helper。当前 persistent batch 主线没有直接用到，
# 保留它主要是为了旧路径/局部表达式复用。
@inline function control_bound_from_x(w, dims, active_u, ui_bound, i, j, sign)
    return active_u * (sign * x_ui(w, dims, i, j) - ui_bound)
end

# 当前 batch 主线会直接复用的 indexed helper：
# 给 JuMP 参数化模型构造绳索-绳索间距约束项。
Base.@noinline function pair_term_indexed(x, base, stride, Rl, ra_i1, ra_i2, ra_i3, ra_j1, ra_j2, ra_j3, cl0, rq, eps_margin, kc, num_dis, i, j)
    frac = kc / num_dis
    ioff = base + (i - 1) * stride
    joff = base + (j - 1) * stride
    pib_i1 = ra_i1 + frac * cl0 * (Rl[1, 1] * x[ioff + 1] + Rl[2, 1] * x[ioff + 2] + Rl[3, 1] * x[ioff + 3])
    pib_i2 = ra_i2 + frac * cl0 * (Rl[1, 2] * x[ioff + 1] + Rl[2, 2] * x[ioff + 2] + Rl[3, 2] * x[ioff + 3])
    pib_j1 = ra_j1 + frac * cl0 * (Rl[1, 1] * x[joff + 1] + Rl[2, 1] * x[joff + 2] + Rl[3, 1] * x[joff + 3])
    pib_j2 = ra_j2 + frac * cl0 * (Rl[1, 2] * x[joff + 1] + Rl[2, 2] * x[joff + 2] + Rl[3, 2] * x[joff + 3])
    return ((kc / num_dis) * 4.0 * rq)^2 + eps_margin - ((pib_i1 - pib_j1)^2 + (pib_i2 - pib_j2)^2)
end

# 当前 batch 主线会直接复用的 indexed helper：
# 给 JuMP 参数化模型构造几何上下界项。
Base.@noinline function gio_terms_indexed(x, base, stride, Rl, ra1, ra2, ra3, cl0, rl, i)
    ioff = base + (i - 1) * stride
    ra_norm = sqrt(ra1^2 + ra2^2 + ra3^2) + 1e-9
    pib1 = ra1 + cl0 * (Rl[1, 1] * x[ioff + 1] + Rl[2, 1] * x[ioff + 2] + Rl[3, 1] * x[ioff + 3])
    pib2 = ra2 + cl0 * (Rl[1, 2] * x[ioff + 1] + Rl[2, 2] * x[ioff + 2] + Rl[3, 2] * x[ioff + 3])
    ei_pib = (ra1 / ra_norm) * pib1 + (ra2 / ra_norm) * pib2
    return ei_pib - (rl + cl0), -rl - ei_pib
end

# 当前 batch 主线会直接复用的 indexed helper：
# 给 JuMP 参数化模型构造每架无人机的推力约束项。
Base.@noinline function thrust_terms_indexed(x, base, stride, Rl, wl, awl, ra1, ra2, ra3, cl0, mq, ml, g, ul1, ul2, ul3, i)
    off = base + (i - 1) * stride
    d1 = x[off + 1]
    d2 = x[off + 2]
    d3 = x[off + 3]
    w1 = x[off + 4]
    w2 = x[off + 5]
    w3 = x[off + 6]
    dw1 = x[off + 7]
    dw2 = x[off + 8]
    dw3 = x[off + 9]
    t = x[off + 13]

    cwr1, cwr2, cwr3 = cross3(wl[1], wl[2], wl[3], ra1, ra2, ra3)
    cwcr1, cwcr2, cwcr3 = cross3(wl[1], wl[2], wl[3], cwr1, cwr2, cwr3)
    car1, car2, car3 = cross3(awl[1], awl[2], awl[3], ra1, ra2, ra3)
    rotb1 = cwcr1 + car1
    rotb2 = cwcr2 + car2
    rotb3 = cwcr3 + car3
    rot1 = Rl[1, 1] * rotb1 + Rl[1, 2] * rotb2 + Rl[1, 3] * rotb3
    rot2 = Rl[2, 1] * rotb1 + Rl[2, 2] * rotb2 + Rl[2, 3] * rotb3
    rot3 = Rl[3, 1] * rotb1 + Rl[3, 2] * rotb2 + Rl[3, 3] * rotb3

    cdd1, cdd2, cdd3 = cross3(dw1, dw2, dw3, d1, d2, d3)
    cwd1, cwd2, cwd3 = cross3(w1, w2, w3, d1, d2, d3)
    cwwd1, cwwd2, cwwd3 = cross3(w1, w2, w3, cwd1, cwd2, cwd3)
    cable1 = cl0 * (cdd1 + cwwd1)
    cable2 = cl0 * (cdd2 + cwwd2)
    cable3 = cl0 * (cdd3 + cwwd3)

    f1 = mq * (ul1 / ml + rot1 + cable1) + d1 * t
    f2 = mq * (ul2 / ml + rot2 + cable2) + d2 * t
    # Match inequality_residual: no extra gravity term after expanding al3 + g.
    f3 = mq * (ul3 / ml + rot3 + cable3) + d3 * t
    return f1, f2, f3
end

# 当前主线会直接依赖的目标函数定义。
# Python 侧也会用同一数学形式做对照/快照导出。
function objective(w, params, dims)
    xl, ul, xc, uc = unpack_w(w, dims)
    active_u = 1.0 - Float64(params["is_terminal"])
    rho_lx = Float64(params["rho_lx"])
    rho_lu = Float64(params["rho_lu"])
    rho_ix = Float64(params["rho_ix"])
    rho_iu = Float64(params["rho_iu"])

    xl_ideal = as_vec(params["xl_ideal"])
    ul_ideal = as_vec(params["ul_ideal"])
    y_xl = as_vec(params["y_xl"])
    y_ul = as_vec(params["y_ul"])
    xc_ideal = as_mat(params["xc_ideal"])
    uc_ideal = as_mat(params["uc_ideal"])
    y_xc = as_mat(params["y_xc"])
    y_uc = as_mat(params["y_uc"])

    res_xl = xl .- xl_ideal .+ y_xl ./ (rho_lx + 1e-6)
    res_ul = ul .- ul_ideal .+ y_ul ./ (rho_lu + 1e-6)
    res_xc = xc .- xc_ideal .+ y_xc ./ (rho_ix + 1e-6)
    res_uc = uc .- uc_ideal .+ y_uc ./ (rho_iu + 1e-6)

    cost = 0.5 * rho_lx * sum(abs2, res_xl)
    cost += active_u * 0.5 * rho_lu * sum(abs2, res_ul)
    cost += 0.5 * rho_ix * sum(abs2, res_xc)
    cost += active_u * 0.5 * rho_iu * sum(abs2, res_uc)
    return cost
end

# 当前主线会直接依赖的等式残差定义。
# 用来做单步诊断、benchmark 统计和解质量核对。
function equality_residual(w, params, dims)
    xl, ul, xc, _ = unpack_w(w, dims)
    active_u = 1.0 - Float64(params["is_terminal"])
    ql = xl[7:10]
    di = xc[:, 1:3]
    Rl = q_to_rotation(ql)
    ti = xc[:, 13]
    fi_inertial = di .* ti
    fi_body = (transpose(Rl) * transpose(fi_inertial))'
    Pt = as_mat(params["Pt"])
    wrench_generated = Pt * vec(fi_body')
    Fl_body = transpose(Rl) * ul[1:3]
    wrench_target = vcat(Fl_body, ul[4:6])

    out = Any[]
    push!(out, sum(abs2, ql) - 1.0)
    for i in axes(di, 1)
        row_norm_sq = di[i, 1]^2 + di[i, 2]^2 + di[i, 3]^2
        push!(out, row_norm_sq - 1.0)
    end
    append!(out, vec(active_u .* (wrench_generated .- wrench_target)))
    return out
end

# 单独取某一维等式残差。更偏诊断/兼容接口，当前主线一般不单独调它。
function equality_component(w, params, dims, idx)
    return equality_residual(w, params, dims)[idx]
end

# 当前主线会直接依赖的不等式残差定义。
# 这里把避障、张力、控制边界、推力边界等全部按统一顺序展开。
function inequality_residual(w, params, dims)
    xl, ul, xc, uc = unpack_w(w, dims)
    _, _, _, nui, nq, num_dis = dims
    active_u = 1.0 - Float64(params["is_terminal"])
    eps_margin = 1e-2

    pl = xl[1:3]
    ql = xl[7:10]
    wl = xl[11:13]
    Rl = q_to_rotation(ql)
    di = xc[:, 1:3]
    wi = xc[:, 4:6]
    dwi = xc[:, 7:9]
    ti = xc[:, 13]
    ra = transpose(as_mat(params["ra"]))
    pob1 = as_vec(params["pob1"])
    pob2 = as_vec(params["pob2"])
    ro = Float64(params["ro"])
    rq = Float64(params["rq"])
    rl = Float64(params["rl"])
    cl0 = Float64(params["cl0"])
    t_min = Float64(params["t_min"])
    t_max = Float64(params["t_max"])
    ui_bound = Float64(params["ui_bound"])
    g = Float64(params["g"])
    ml = Float64(params["ml"])
    mq = Float64(params["mq"])
    fmax = Float64(params["fmax"])
    Jl = as_mat(params["Jl"])
    Jl_inv = as_mat(params["Jl_inv"])

    ineq = Any[]

    safe_r_l = ro + 0.5 * rq
    dist_l_obs1 = sum((pl[1:2] .- pob1[1:2]) .^ 2)
    dist_l_obs2 = sum((pl[1:2] .- pob2[1:2]) .^ 2)
    push!(ineq, safe_r_l^2 + eps_margin - dist_l_obs1)
    push!(ineq, safe_r_l^2 + eps_margin - dist_l_obs2)

    pi_mat = Array{Any}(undef, nq, 3)
    for i in 1:nq
        tmp = pl .+ Rl * ra[i, :] .+ cl0 * di[i, :]
        for j in 1:3
            pi_mat[i, j] = tmp[j]
        end
    end
    safe_r_q = ro + 2.0 * rq
    obs1_terms = Any[]
    obs2_terms = Any[]
    for i in 1:nq
        push!(obs1_terms, safe_r_q^2 + eps_margin - sum((pi_mat[i, 1:2] .- pob1[1:2]) .^ 2))
        push!(obs2_terms, safe_r_q^2 + eps_margin - sum((pi_mat[i, 1:2] .- pob2[1:2]) .^ 2))
    end
    append!(ineq, obs1_terms)
    append!(ineq, obs2_terms)

    pair_terms = Any[]
    gio_upper = Any[]
    gio_lower = Any[]
    for kc in 1:num_dis
        frac = kc / num_dis
        pib = Array{Any}(undef, nq, 3)
        for i in 1:nq
            tmp = ra[i, :] .+ frac * cl0 * (transpose(Rl) * di[i, :])
            for j in 1:3
                pib[i, j] = tmp[j]
            end
        end
        min_pair_d2 = (frac * 4.0 * rq)^2 + eps_margin
        for i in 1:nq
            for j in i+1:nq
                dij_sq = sum((pib[i, 1:2] .- pib[j, 1:2]) .^ 2)
                push!(pair_terms, min_pair_d2 - dij_sq)
            end
        end
        if kc == num_dis
            for i in 1:nq
                ra_i = ra[i, :]
                ra_norm = sqrt(sum(ra_i .^ 2)) + 1e-9
                ei_i = ra_i ./ ra_norm
                ei_pib = sum(ei_i[1:2] .* pib[i, 1:2])
                push!(gio_upper, ei_pib - (rl + cl0))
                push!(gio_lower, -rl - ei_pib)
            end
        end
    end
    append!(ineq, gio_upper)
    append!(ineq, gio_lower)
    append!(ineq, pair_terms)

    for i in 1:nq
        push!(ineq, t_min - ti[i])
    end
    for i in 1:nq
        push!(ineq, ti[i] - t_max)
    end
    for i in 1:nq
        for j in 1:nui
            push!(ineq, active_u * (uc[i, j] - ui_bound))
        end
    end
    for i in 1:nq
        for j in 1:nui
            push!(ineq, active_u * (-uc[i, j] - ui_bound))
        end
    end

    Fl = ul[1:3]
    Ml = ul[4:6]
    al = Any[Fl[1] / ml, Fl[2] / ml, Fl[3] / ml - g]
    awl = Jl_inv * (Ml - cross(wl, Jl * wl))
    thrust_upper = Any[]
    thrust_lower = Any[]
    for i in 1:nq
        rot_term = Rl * (cross(wl, cross(wl, ra[i, :])) + cross(awl, ra[i, :]))
        cable_term = cl0 * (cross(dwi[i, :], di[i, :]) + cross(wi[i, :], cross(wi[i, :], di[i, :])))
        fi = Any[
            mq * (al[1] + rot_term[1] + cable_term[1]) + di[i, 1] * ti[i],
            mq * (al[2] + rot_term[2] + cable_term[2]) + di[i, 2] * ti[i],
            mq * (al[3] + rot_term[3] + cable_term[3] + g) + di[i, 3] * ti[i],
        ]
        thrust_sq = fi[1]^2 + fi[2]^2 + fi[3]^2
        push!(thrust_upper, active_u * (thrust_sq - fmax^2))
        push!(thrust_lower, active_u * (eps_margin - thrust_sq))
    end
    append!(ineq, thrust_upper)
    append!(ineq, thrust_lower)
    return ineq
end

# 单独取某一维不等式残差。主要给诊断/兼容接口用。
function inequality_component(w, params, dims, idx)
    return inequality_residual(w, params, dims)[idx]
end

# 兼容旧拼写的别名接口。
# 名字里 residual 拼成了 resual，当前不建议再把它当新入口使用。
function inequality_resual_component(w, params, dims, idx)
    return inequality_component(w, params, dims, idx)
end

# 可先跳过：ExaModels 路径下的等式约束构造。
# 当前 persistent Julia 主线走 JuMP/MadNLP 参数化模型，不直接走这里。
function add_native_equality_constraints!(core, x, step::StepData)
    params = step.params
    dims = step.dims
    nxl, _, _, _, nq, _ = dims
    active_u = 1.0 - Float64(params["is_terminal"])
    Pt = as_mat(params["Pt"])
    Rl = q_to_rotation_from_x(x, dims)

    ExaModels.constraint(
        core,
        x_xl(x, 7)^2 + x_xl(x, 8)^2 + x_xl(x, 9)^2 + x_xl(x, 10)^2 - 1.0;
        lcon = 0.0,
        ucon = 0.0,
    )

    ExaModels.constraint(
        core,
        x_xi(x, dims, i, 1)^2 + x_xi(x, dims, i, 2)^2 + x_xi(x, dims, i, 3)^2 - 1.0 for i in 1:nq;
        lcon = 0.0,
        ucon = 0.0,
    )

    fi_body = Array{Any}(undef, nq, 3)
    for i in 1:nq
        fx = x_xi(x, dims, i, 1) * x_xi(x, dims, i, 13)
        fy = x_xi(x, dims, i, 2) * x_xi(x, dims, i, 13)
        fz = x_xi(x, dims, i, 3) * x_xi(x, dims, i, 13)
        fi_body[i, 1] = Rl[1, 1] * fx + Rl[2, 1] * fy + Rl[3, 1] * fz
        fi_body[i, 2] = Rl[1, 2] * fx + Rl[2, 2] * fy + Rl[3, 2] * fz
        fi_body[i, 3] = Rl[1, 3] * fx + Rl[2, 3] * fy + Rl[3, 3] * fz
    end

    wrench_generated = Any[]
    for r in 1:size(Pt, 1)
        expr = 0.0
        col = 1
        for i in 1:nq
            for j in 1:3
                expr += Pt[r, col] * fi_body[i, j]
                col += 1
            end
        end
        push!(wrench_generated, expr)
    end
    Fl_body = Any[
        Rl[1, 1] * x_ul(x, nxl, 1) + Rl[2, 1] * x_ul(x, nxl, 2) + Rl[3, 1] * x_ul(x, nxl, 3),
        Rl[1, 2] * x_ul(x, nxl, 1) + Rl[2, 2] * x_ul(x, nxl, 2) + Rl[3, 2] * x_ul(x, nxl, 3),
        Rl[1, 3] * x_ul(x, nxl, 1) + Rl[2, 3] * x_ul(x, nxl, 2) + Rl[3, 3] * x_ul(x, nxl, 3),
    ]
    wrench_target = Any[Fl_body[1], Fl_body[2], Fl_body[3], x_ul(x, nxl, 4), x_ul(x, nxl, 5), x_ul(x, nxl, 6)]

    for r in 1:6
        ExaModels.constraint(
            core,
            active_u * (wrench_generated[r] - wrench_target[r]);
            lcon = 0.0,
            ucon = 0.0,
        )
    end
end

# 可先跳过：ExaModels 路径下的不等式约束构造。
# 保留它主要是为了旧实验/兼容 ExaModels，不是当前主线热路径。
function add_native_inequality_constraints!(core, x, step::StepData; include_thrust::Bool = true)
    params = step.params
    dims = step.dims
    _, _, _, nui, nq, num_dis = dims
    xl, ul, xc, uc = unpack_w(x, dims)
    pl = xl[1:3]
    ql = xl[7:10]
    wl = xl[11:13]
    di = xc[:, 1:3]
    wi = xc[:, 4:6]
    dwi = xc[:, 7:9]
    ti = xc[:, 13]
    Rl = q_to_rotation(ql)

    active_u = 1.0 - Float64(params["is_terminal"])
    eps_margin = 1e-2

    ra = transpose(as_mat(params["ra"]))
    ra_p = ExaModels.parameter(core, ra)
    pob1 = as_vec(params["pob1"])
    pob2 = as_vec(params["pob2"])
    ro = Float64(params["ro"])
    rq = Float64(params["rq"])
    rl = Float64(params["rl"])
    cl0 = Float64(params["cl0"])
    t_min = Float64(params["t_min"])
    t_max = Float64(params["t_max"])
    ui_bound = Float64(params["ui_bound"])
    g = Float64(params["g"])
    ml = Float64(params["ml"])
    mq = Float64(params["mq"])
    fmax = Float64(params["fmax"])
    Jl = as_mat(params["Jl"])
    Jl_inv = as_mat(params["Jl_inv"])

    safe_r_l = ro + 0.5 * rq
    ExaModels.constraint(
        core,
        safe_r_l^2 + eps_margin - ((pl[1] - pob1[1])^2 + (pl[2] - pob1[2])^2);
        lcon = -Inf,
        ucon = 0.0,
    )
    ExaModels.constraint(
        core,
        safe_r_l^2 + eps_margin - ((pl[1] - pob2[1])^2 + (pl[2] - pob2[2])^2);
        lcon = -Inf,
        ucon = 0.0,
    )

    safe_r_q = ro + 2.0 * rq
    ExaModels.constraint(
        core,
        safe_r_q^2 + eps_margin -
        ((pl[1] + (Rl[1, 1] * ra_p[i, 1] + Rl[1, 2] * ra_p[i, 2] + Rl[1, 3] * ra_p[i, 3]) + cl0 * x_xi(x, dims, i, 1) - pob1[1])^2 +
         (pl[2] + (Rl[2, 1] * ra_p[i, 1] + Rl[2, 2] * ra_p[i, 2] + Rl[2, 3] * ra_p[i, 3]) + cl0 * x_xi(x, dims, i, 2) - pob1[2])^2)
        for i in 1:nq;
        lcon = -Inf,
        ucon = 0.0,
    )
    ExaModels.constraint(
        core,
        safe_r_q^2 + eps_margin -
        ((pl[1] + (Rl[1, 1] * ra_p[i, 1] + Rl[1, 2] * ra_p[i, 2] + Rl[1, 3] * ra_p[i, 3]) + cl0 * x_xi(x, dims, i, 1) - pob2[1])^2 +
         (pl[2] + (Rl[2, 1] * ra_p[i, 1] + Rl[2, 2] * ra_p[i, 2] + Rl[2, 3] * ra_p[i, 3]) + cl0 * x_xi(x, dims, i, 2) - pob2[2])^2)
        for i in 1:nq;
        lcon = -Inf,
        ucon = 0.0,
    )

    ExaModels.constraint(
        core,
        ((kc / num_dis) * 4.0 * rq)^2 + eps_margin -
        (
            (
                (ra_p[i, 1] + (kc / num_dis) * cl0 * (Rl[1, 1] * x_xi(x, dims, i, 1) + Rl[2, 1] * x_xi(x, dims, i, 2) + Rl[3, 1] * x_xi(x, dims, i, 3))) -
                (ra_p[j, 1] + (kc / num_dis) * cl0 * (Rl[1, 1] * x_xi(x, dims, j, 1) + Rl[2, 1] * x_xi(x, dims, j, 2) + Rl[3, 1] * x_xi(x, dims, j, 3)))
            )^2 +
            (
                (ra_p[i, 2] + (kc / num_dis) * cl0 * (Rl[1, 2] * x_xi(x, dims, i, 1) + Rl[2, 2] * x_xi(x, dims, i, 2) + Rl[3, 2] * x_xi(x, dims, i, 3))) -
                (ra_p[j, 2] + (kc / num_dis) * cl0 * (Rl[1, 2] * x_xi(x, dims, j, 1) + Rl[2, 2] * x_xi(x, dims, j, 2) + Rl[3, 2] * x_xi(x, dims, j, 3)))
            )^2
        )
        for kc in 1:num_dis, i in 1:nq, j in 1:nq if j > i;
        lcon = -Inf,
        ucon = 0.0,
    )

    ExaModels.constraint(
        core,
        (
            ra_p[i, 1] / (sqrt(ra_p[i, 1]^2 + ra_p[i, 2]^2 + ra_p[i, 3]^2) + 1e-9)
        ) * (
            ra_p[i, 1] + cl0 * (Rl[1, 1] * x_xi(x, dims, i, 1) + Rl[2, 1] * x_xi(x, dims, i, 2) + Rl[3, 1] * x_xi(x, dims, i, 3))
        ) +
        (
            ra_p[i, 2] / (sqrt(ra_p[i, 1]^2 + ra_p[i, 2]^2 + ra_p[i, 3]^2) + 1e-9)
        ) * (
            ra_p[i, 2] + cl0 * (Rl[1, 2] * x_xi(x, dims, i, 1) + Rl[2, 2] * x_xi(x, dims, i, 2) + Rl[3, 2] * x_xi(x, dims, i, 3))
        ) -
        (rl + cl0)
        for i in 1:nq;
        lcon = -Inf,
        ucon = 0.0,
    )

    ExaModels.constraint(
        core,
        -rl -
        (
            (
                ra_p[i, 1] / (sqrt(ra_p[i, 1]^2 + ra_p[i, 2]^2 + ra_p[i, 3]^2) + 1e-9)
            ) * (
                ra_p[i, 1] + cl0 * (Rl[1, 1] * x_xi(x, dims, i, 1) + Rl[2, 1] * x_xi(x, dims, i, 2) + Rl[3, 1] * x_xi(x, dims, i, 3))
            ) +
            (
                ra_p[i, 2] / (sqrt(ra_p[i, 1]^2 + ra_p[i, 2]^2 + ra_p[i, 3]^2) + 1e-9)
            ) * (
                ra_p[i, 2] + cl0 * (Rl[1, 2] * x_xi(x, dims, i, 1) + Rl[2, 2] * x_xi(x, dims, i, 2) + Rl[3, 2] * x_xi(x, dims, i, 3))
            )
        )
        for i in 1:nq;
        lcon = -Inf,
        ucon = 0.0,
    )

    ExaModels.constraint(core, t_min - x_xi(x, dims, i, 13) for i in 1:nq; lcon = -Inf, ucon = 0.0)
    ExaModels.constraint(core, x_xi(x, dims, i, 13) - t_max for i in 1:nq; lcon = -Inf, ucon = 0.0)
    ExaModels.constraint(core, active_u * (x_ui(x, dims, i, j) - ui_bound) for i in 1:nq, j in 1:nui; lcon = -Inf, ucon = 0.0)
    ExaModels.constraint(core, active_u * (-x_ui(x, dims, i, j) - ui_bound) for i in 1:nq, j in 1:nui; lcon = -Inf, ucon = 0.0)

    if include_thrust
        awl = Jl_inv * (ul[4:6] - cross(wl, Jl * wl))
        ExaModels.constraint(
            core,
            active_u * (
                begin
                    f1, f2, f3 = thrust_terms_from_x(
                        x, dims, Rl, wl, awl, ra_p[i, 1], ra_p[i, 2], ra_p[i, 3], cl0, mq, ml, g, ul[1], ul[2], ul[3], i
                    )
                    f1^2 + f2^2 + f3^2 - fmax^2
                end
            )
            for i in 1:nq;
            lcon = -Inf,
            ucon = 0.0,
        )
        ExaModels.constraint(
            core,
            active_u * (
                begin
                    f1, f2, f3 = thrust_terms_from_x(
                        x, dims, Rl, wl, awl, ra_p[i, 1], ra_p[i, 2], ra_p[i, 3], cl0, mq, ml, g, ul[1], ul[2], ul[3], i
                    )
                    eps_margin - (f1^2 + f2^2 + f3^2)
                end
            )
            for i in 1:nq;
            lcon = -Inf,
            ucon = 0.0,
        )
    end
end

# 当前 batch 主线必经的数据入口之一。
# Python 发来的 step snapshot / stacked JSON 最终都会被整理成 StepData。
function step_data_from_json(data)
    dims = Tuple(Int.(data["step"]["dims"]))
    step = data["step"]
    x_init = as_vec(step["x_init"])
    x_orig =
        haskey(step, "original_step_reference") ? as_vec(step["original_step_reference"]) : copy(x_init)
    init_ref = get(step, "init_reference_values", nothing)
    orig_ref = get(step, "original_reference_values", nothing)
    eq_dim = haskey(data["meta"], "eq_dim") ? Int(data["meta"]["eq_dim"]) : 0
    ineq_dim = haskey(data["meta"], "ineq_dim") ? Int(data["meta"]["ineq_dim"]) : 0
    return StepData(
        data["meta"],
        dims,
        step["params_t"],
        x_init,
        x_orig,
        init_ref === nothing ? NaN : Float64(init_ref["objective"]),
        init_ref === nothing ? zeros(Float64, eq_dim) : as_vec(init_ref["eq_vec"]),
        init_ref === nothing ? zeros(Float64, ineq_dim) : as_vec(init_ref["ineq_vec"]),
        orig_ref === nothing ? NaN : Float64(orig_ref["objective"]),
        orig_ref === nothing ? zeros(Float64, eq_dim) : as_vec(orig_ref["eq_vec"]),
        orig_ref === nothing ? zeros(Float64, ineq_dim) : as_vec(orig_ref["ineq_vec"]),
    )
end

# 文件路径包装层，主要给单步离线调试用。当前 persistent worker 主线更常直接走内存 payload。
function load_step_data(path::String)
    data = JSON3.read(read(path, String))
    return step_data_from_json(data)
end

# 可先跳过：ExaModels 建模入口。
# 它不是当前 persistent Julia + JuMP/MadNLP 主线的一部分，保留原因主要是历史实验和兼容验证。
function build_examodel(step::StepData; include_ineq::Bool = true, include_thrust::Bool = true)
    core = ExaCore()
    x = variable(core, length(step.x_init); start = step.x_init)
    ExaModels.objective(core, objective(x, step.params, step.dims))
    add_native_equality_constraints!(core, x, step)
    if include_ineq
        add_native_inequality_constraints!(core, x, step; include_thrust = include_thrust)
    end
    model = ExaModel(core; prod = true)
    return model, x
end

end
