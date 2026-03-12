import Pkg
const _ROOT = dirname(@__DIR__)
Pkg.activate(_ROOT)

using JSON3
using LinearAlgebra

include(joinpath(_ROOT, "step_model.jl"))
using .SubP2StepModel

function native_family_values(w, step::StepData)
    params = step.params
    dims = step.dims
    nxl, nul, nxi, nui, nq, num_dis = dims

    xl, ul, xc, uc = SubP2StepModel.unpack_w(w, dims)
    active_u = 1.0 - Float64(params["is_terminal"])
    pob1 = as_vec(params["pob1"])
    pob2 = as_vec(params["pob2"])
    ra = transpose(as_mat(params["ra"]))
    ro = Float64(params["ro"])
    rq = Float64(params["rq"])
    cl0 = Float64(params["cl0"])
    rl = Float64(params["rl"])
    t_min = Float64(params["t_min"])
    t_max = Float64(params["t_max"])
    ui_bound = Float64(params["ui_bound"])
    eps_margin = 1e-2
    g = Float64(params["g"])
    ml = Float64(params["ml"])
    mq = Float64(params["mq"])
    fmax = Float64(params["fmax"])
    Pt = as_mat(params["Pt"])
    Jl = as_mat(params["Jl"])
    Jl_inv = as_mat(params["Jl_inv"])

    pl = xl[1:3]
    ql = xl[7:10]
    wl = xl[11:13]
    Rl = SubP2StepModel.q_to_rotation(ql)
    di = xc[:, 1:3]
    wi = xc[:, 4:6]
    dwi = xc[:, 7:9]
    ti = xc[:, 13]

    out = Dict{String, Vector{Float64}}()

    # equalities
    out["eq_load_quat"] = [sum(abs2, ql) - 1.0]
    out["eq_cable_norm"] = [di[i, 1]^2 + di[i, 2]^2 + di[i, 3]^2 - 1.0 for i in 1:nq]

    fi_body = zeros(Float64, nq, 3)
    for i in 1:nq
        fx = di[i, 1] * ti[i]
        fy = di[i, 2] * ti[i]
        fz = di[i, 3] * ti[i]
        fi_body[i, 1] = Rl[1, 1] * fx + Rl[2, 1] * fy + Rl[3, 1] * fz
        fi_body[i, 2] = Rl[1, 2] * fx + Rl[2, 2] * fy + Rl[3, 2] * fz
        fi_body[i, 3] = Rl[1, 3] * fx + Rl[2, 3] * fy + Rl[3, 3] * fz
    end
    wrench_generated = zeros(Float64, 6)
    for r in 1:6
        col = 1
        for i in 1:nq, j in 1:3
            wrench_generated[r] += Pt[r, col] * fi_body[i, j]
            col += 1
        end
    end
    wrench_target = vcat(transpose(Rl) * ul[1:3], ul[4:6])
    out["eq_wrench"] = vec(active_u .* (wrench_generated .- wrench_target))

    # inequalities
    safe_r_l = ro + 0.5 * rq
    out["ineq_load_obs"] = Float64[
        safe_r_l^2 + eps_margin - ((xl[1] - pob1[1])^2 + (xl[2] - pob1[2])^2),
        safe_r_l^2 + eps_margin - ((xl[1] - pob2[1])^2 + (xl[2] - pob2[2])^2),
    ]

    safe_r_q = ro + 2.0 * rq
    pi1 = zeros(Float64, nq)
    pi2 = zeros(Float64, nq)
    for i in 1:nq
        pi1[i] = xl[1] + (Rl[1, 1] * ra[i, 1] + Rl[1, 2] * ra[i, 2] + Rl[1, 3] * ra[i, 3]) + cl0 * di[i, 1]
        pi2[i] = xl[2] + (Rl[2, 1] * ra[i, 1] + Rl[2, 2] * ra[i, 2] + Rl[2, 3] * ra[i, 3]) + cl0 * di[i, 2]
    end
    out["ineq_cable_obs"] = vcat(
        [safe_r_q^2 + eps_margin - ((pi1[i] - pob1[1])^2 + (pi2[i] - pob1[2])^2) for i in 1:nq],
        [safe_r_q^2 + eps_margin - ((pi1[i] - pob2[1])^2 + (pi2[i] - pob2[2])^2) for i in 1:nq],
    )

    base = nxl + nul
    stride = nxi + nui
    pair_terms = Float64[]
    for kc in 1:num_dis
        for i in 1:nq
            for j in i+1:nq
                push!(pair_terms, SubP2StepModel.pair_term_indexed(w, base, stride, Rl, ra[i, 1], ra[i, 2], ra[i, 3], ra[j, 1], ra[j, 2], ra[j, 3], cl0, rq, eps_margin, kc, num_dis, i, j))
            end
        end
    end
    out["ineq_pair"] = pair_terms

    gio_u = Float64[]
    gio_l = Float64[]
    for i in 1:nq
        u, l = SubP2StepModel.gio_terms_indexed(w, base, stride, Rl, ra[i, 1], ra[i, 2], ra[i, 3], cl0, rl, i)
        push!(gio_u, u)
        push!(gio_l, l)
    end
    out["ineq_gio_upper"] = gio_u
    out["ineq_gio_lower"] = gio_l

    out["ineq_tension_lower"] = [t_min - ti[i] for i in 1:nq]
    out["ineq_tension_upper"] = [ti[i] - t_max for i in 1:nq]
    control_upper = Float64[]
    control_lower = Float64[]
    for i in 1:nq, j in 1:nui
        push!(control_upper, SubP2StepModel.control_bound_from_x(w, dims, active_u, ui_bound, i, j, 1.0))
        push!(control_lower, SubP2StepModel.control_bound_from_x(w, dims, active_u, ui_bound, i, j, -1.0))
    end
    out["ineq_control_upper"] = control_upper
    out["ineq_control_lower"] = control_lower

    Jw = Jl * wl
    cwJ = collect(cross(wl, Jw))
    awl = Jl_inv * (ul[4:6] - cwJ)
    al = [ul[1] / ml, ul[2] / ml, ul[3] / ml - g]
    thrust_upper = Float64[]
    thrust_lower = Float64[]
    for i in 1:nq
        f1, f2, f3 = SubP2StepModel.thrust_terms_indexed(
            w, base, stride, Rl, wl, awl, ra[i, 1], ra[i, 2], ra[i, 3], cl0, mq, ml, g, ul[1], ul[2], ul[3], i
        )
        thrust_sq = f1^2 + f2^2 + f3^2
        push!(thrust_upper, active_u * (thrust_sq - fmax^2))
        push!(thrust_lower, active_u * (eps_margin - thrust_sq))
    end
    out["ineq_thrust_upper"] = thrust_upper
    out["ineq_thrust_lower"] = thrust_lower
    return out
end

function ref_family_values(w, step::StepData)
    eq = Float64.(SubP2StepModel.equality_residual(w, step.params, step.dims))
    ineq = Float64.(SubP2StepModel.inequality_residual(w, step.params, step.dims))
    out = Dict{String, Vector{Float64}}()
    out["eq_load_quat"] = eq[1:1]
    out["eq_cable_norm"] = eq[2:5]
    out["eq_wrench"] = eq[6:11]
    p = 1
    out["ineq_load_obs"] = ineq[p:p+1]; p += 2
    out["ineq_cable_obs"] = ineq[p:p+7]; p += 8
    out["ineq_gio_upper"] = ineq[p:p+3]; p += 4
    out["ineq_gio_lower"] = ineq[p:p+3]; p += 4
    out["ineq_pair"] = ineq[p:p+11]; p += 12
    out["ineq_tension_lower"] = ineq[p:p+3]; p += 4
    out["ineq_tension_upper"] = ineq[p:p+3]; p += 4
    out["ineq_control_upper"] = ineq[p:p+15]; p += 16
    out["ineq_control_lower"] = ineq[p:p+15]; p += 16
    out["ineq_thrust_upper"] = ineq[p:p+3]; p += 4
    out["ineq_thrust_lower"] = ineq[p:p+3]
    return out
end

function audit_point(name::String, w, step::StepData)
    native = native_family_values(w, step)
    ref = ref_family_values(w, step)
    println("[family-audit] point=", name)
    worst = 0.0
    for key in [
        "eq_load_quat","eq_cable_norm","eq_wrench",
        "ineq_load_obs","ineq_cable_obs","ineq_gio_upper","ineq_gio_lower",
        "ineq_pair","ineq_tension_lower","ineq_tension_upper",
        "ineq_control_upper","ineq_control_lower",
        "ineq_thrust_upper","ineq_thrust_lower",
    ]
        @assert length(native[key]) == length(ref[key])
        d = maximum(abs.(native[key] .- ref[key]))
        worst = max(worst, d)
        println("  ", rpad(key, 22), " max_abs_diff = ", d)
    end
    println("  ", rpad("overall", 22), " max_abs_diff = ", worst)
end

function main()
    if length(ARGS) < 1
        error("usage: julia julia_subp2/archive/audit_native_families.jl <snapshot.json>")
    end
    step = SubP2StepModel.load_step_data(ARGS[1])
    audit_point("x_init", step.x_init, step)
    audit_point("x_orig", step.x_orig, step)
end

main()
