from jax import numpy as np

import jax

from enum import Enum

from functools import partial

from .linalg_helpers import project_psd_cone


class LinearSystemFormulation(Enum):
    AUTODIFF = 0
    STABLE_DIRECT_4x4 = 1
    SYMMETRIC_DIRECT_4x4 = 2
    SYMMETRIC_INDIRECT_3x3 = 3
    SYMMETRIC_INDIRECT_2x2 = 4


@partial(
    jax.jit,
    static_argnames=(
        "f",
        "c",
        "g",
        "lin_sys_formulation",
        "lin_sys_solver",
        "psd_use_lapack",
        "psd_iterate",
        "print_logs",
        "trace_length",
    ),
)
def solve(
    *,
    f,
    c,
    g,
    ws_x,
    ws_s,
    ws_y,
    ws_z,
    max_iterations=100,
    max_kkt_violation=1e-6,
    lin_sys_formulation=LinearSystemFormulation.SYMMETRIC_INDIRECT_2x2,
    lin_sys_solver=np.linalg.solve,
    psd_use_lapack=True,
    psd_iterate=True,
    tau_min=0.995,
    mu_min=1e-12,
    min_delta=1e-9,
    gamma_y=1e-6,
    gamma_z=1e-6,
    armijo_factor=1e-4,
    feasibility_armijo_factor=1e-4,
    line_search_factor=0.5,
    line_search_min_step_size=1e-6,
    positivity_floor=1e-12,
    soft_stop_enabled=False,
    soft_mu_tol=0.0,
    soft_comp_tol=0.0,
    soft_dual_tol=0.0,
    soft_eq_tol=0.0,
    soft_ineq_tol=0.0,
    plateau_stop_enabled=False,
    plateau_min_iterations=0,
    plateau_patience=0,
    plateau_phi_tol=0.0,
    plateau_ineq_tol=0.0,
    print_logs=True,
    trace_length=0,
):
    """
    Solves an optimization problem of the form:
        min_x f(x) s.t. (c(x) = 0 and g(x) + s = 0 and s >= 0)

    min_delta determines the minimum regularization on the objective Hessian.
    gamma_y, gamma_z regularize the constraints to ensure the Newton-KKT system is non-singular.
    tau_min is a parameter used in the fraction-to-the-boundary rule.
    line_search_factor determines how much to backtrack at each line search iteration.
    """

    # assert ws_s.min() > 0.0, "ws_s must contain only positive entries."
    # assert ws_z.min() > 0.0, "ws_z must contain only positive entries."

    def split_xsyz_vars(xsyz):
        x_dim = ws_x.shape[0]
        s_dim = ws_s.shape[0]
        z_dim = ws_z.shape[0]

        # assert s_dim == z_dim, "Incompatible shapes of s and z."

        x = xsyz[:x_dim]
        s = xsyz[x_dim : x_dim + s_dim]
        y = xsyz[x_dim + s_dim : -z_dim]
        z = xsyz[-z_dim:]
        return x, s, y, z

    def split_xyz_vars(xyz):
        x_dim = ws_x.shape[0]
        z_dim = ws_z.shape[0]

        x = xyz[:x_dim]
        y = xyz[x_dim:-z_dim]
        z = xyz[-z_dim:]
        return x, y, z

    def split_xy_vars(xy):
        x_dim = ws_x.shape[0]

        x = xy[:x_dim]
        y = xy[x_dim:]
        return x, y

    def combine_xsyz_vars(*, x, s, y, z):
        return np.concatenate([x, s, y, z])

    def barrier_augmented_lagrangian(xsyz, *, mu):
        x, s, y, z = split_xsyz_vars(xsyz)
        return (
            f(x) + np.dot(y, c(x)) + np.dot(z, g(x) + s) - mu * np.log(s).sum()
        )

    def adaptive_mu(*, s, z, iteration):
        # Uses the LOQO rule mentioned in Nocedal & Wright.
        del iteration
        m = s.shape[0]
        dot = np.dot(s, z)
        dot_safe = np.maximum(dot, 1e-16)
        comp_avg = dot_safe / m
        zeta = (s * z).min() * m / dot_safe
        zeta_safe = np.maximum(zeta, 1e-6)
        sigma = 0.1 * np.minimum(0.5 * (1.0 - zeta_safe) / zeta_safe, 2) ** 3
        mu_loqo = sigma * comp_avg
        mu_aggressive = np.sqrt(comp_avg) * comp_avg
        return np.minimum(mu_loqo, mu_aggressive)

    def get_rho(*, x, s, dx, ds, mu):
        # D(merit_function; dx, ds) = D(f; dx) - mu (ds / s) - rho * ||c(x)|| - rho * ||g(x) + s ||
        # rho > (D(f; dx) + k) / (|| (c(x) || + || g(x) + s) || iff D(merit_function; dx) < -k.
        f_slope = jax.grad(f)(x).dot(dx)
        barrier_slope = -(mu / s).dot(ds)
        obj_slope = f_slope + barrier_slope
        d = np.maximum(np.linalg.norm(c(x)) + np.linalg.norm(g(x) + s), 1e-12)
        k = np.maximum(d, 2.0 * np.abs(obj_slope))
        return np.minimum((obj_slope + k) / d, 1e9)

    def merit_function(*, x, s, mu, rho):
        return (
            f(x)
            - mu * np.log(s).sum()
            + rho * np.linalg.norm(c(x))
            + rho * np.linalg.norm(g(x) + s)
        )

    def merit_function_slope(*, x, s, dx, ds, mu, rho):
        return (
            jax.grad(f)(x).dot(dx)
            - (mu / s).dot(ds)
            - rho * np.linalg.norm(c(x))
            - rho * np.linalg.norm(g(x) + s)
        )

    def feasibility_measure(*, x, s):
        return np.linalg.norm(c(x)) + np.linalg.norm(g(x) + s)

    def complementarity_measure(*, s, z, mu):
        return np.abs(s * z - mu).max()

    def compute_search_direction_autodiff(*, x, s, y, z, mu):
        x_dim = x.shape[0]
        s_dim = s.shape[0]
        y_dim = y.shape[0]
        z_dim = z.shape[0]
        xsyz = combine_xsyz_vars(x=x, s=s, y=y, z=z)
        al = partial(barrier_augmented_lagrangian, mu=mu)
        lhs = jax.hessian(al)(xsyz)
        rhs = -jax.grad(al)(xsyz)
        lhs += jax.scipy.linalg.block_diag(
            np.zeros([x_dim, x_dim]),
            np.zeros([s_dim, s_dim]),
            -gamma_y * np.eye(y_dim),
            -gamma_z * np.eye(z_dim),
        )

        lhs = lhs.at[:x_dim, :x_dim].set(
            project_psd_cone(
                lhs[:x_dim, :x_dim],
                delta=min_delta,
                use_lapack=psd_use_lapack,
                iterate=psd_iterate,
            )
        )
        dxsyz = lin_sys_solver(lhs, rhs)
        error = np.linalg.norm(lhs @ dxsyz - rhs)
        return dxsyz, error

    def compute_search_direction_stable_direct_method(*, x, s, y, z, mu):
        x_dim = x.shape[0]
        s_dim = s.shape[0]
        y_dim = y.shape[0]
        z_dim = z.shape[0]

        def al_x(xx):
            return barrier_augmented_lagrangian(
                np.concatenate([xx, s, y, z]), mu=mu
            )

        D1L = jax.grad(al_x)(x)
        D2L = project_psd_cone(
            jax.hessian(al_x)(x),
            delta=min_delta,
            use_lapack=psd_use_lapack,
            iterate=psd_iterate,
        )
        C = jax.jacfwd(c)(x)
        G = jax.jacfwd(g)(x)
        lhs = np.block(
            [
                [D2L, np.zeros([x_dim, s_dim]), C.T, G.T],
                [
                    np.zeros([s_dim, x_dim]),
                    np.diag(z),
                    np.zeros([s_dim, y_dim]),
                    np.diag(s),
                ],
                [
                    C,
                    np.zeros([y_dim, s_dim]),
                    -gamma_y * np.eye(y_dim),
                    np.zeros([y_dim, z_dim]),
                ],
                [
                    G,
                    np.eye(z_dim),
                    np.zeros([z_dim, y_dim]),
                    -gamma_z * np.eye(z_dim),
                ],
            ]
        )

        rhs = -np.concatenate(
            [D1L, s * z - mu * np.ones_like(s), c(x), g(x) + s]
        )

        dxsyz = lin_sys_solver(lhs, rhs)
        error = np.linalg.norm(lhs @ dxsyz - rhs)
        return dxsyz, error

    def compute_search_direction_symmetric_direct_method_4x4(*, x, s, y, z, mu):
        x_dim = x.shape[0]
        s_dim = s.shape[0]
        y_dim = y.shape[0]
        z_dim = z.shape[0]

        def al_x(xx):
            return barrier_augmented_lagrangian(
                np.concatenate([xx, s, y, z]), mu=mu
            )

        D1L = jax.grad(al_x)(x)
        D2L = project_psd_cone(
            jax.hessian(al_x)(x),
            delta=min_delta,
            use_lapack=psd_use_lapack,
            iterate=psd_iterate,
        )
        C = jax.jacfwd(c)(x)
        G = jax.jacfwd(g)(x)
        lhs = np.block(
            [
                [D2L, np.zeros([x_dim, s_dim]), C.T, G.T],
                [
                    np.zeros([s_dim, x_dim]),
                    np.diag(z / s),
                    np.zeros([s_dim, y_dim]),
                    np.eye(s_dim),
                ],
                [
                    C,
                    np.zeros([y_dim, s_dim]),
                    -gamma_y * np.eye(y_dim),
                    np.zeros([y_dim, z_dim]),
                ],
                [
                    G,
                    np.eye(z_dim),
                    np.zeros([z_dim, y_dim]),
                    -gamma_z * np.eye(z_dim),
                ],
            ]
        )

        rhs = -np.concatenate([D1L, z - mu / s, c(x), g(x) + s])

        dxsyz = lin_sys_solver(lhs, rhs)

        error = np.linalg.norm(lhs @ dxsyz - rhs)
        return dxsyz, error

    def compute_search_direction_symmetric_indirect_method_3x3(
        *, x, s, y, z, mu
    ):
        y_dim = y.shape[0]
        z_dim = z.shape[0]

        def al_x(xx):
            return barrier_augmented_lagrangian(
                np.concatenate([xx, s, y, z]), mu=mu
            )

        D1L = jax.grad(al_x)(x)
        D2L = project_psd_cone(
            jax.hessian(al_x)(x),
            delta=min_delta,
            use_lapack=psd_use_lapack,
            iterate=psd_iterate,
        )
        C = jax.jacfwd(c)(x)
        G = jax.jacfwd(g)(x)
        sigma_inv = np.diag(s / z)
        lhs = np.block(
            [
                [D2L, C.T, G.T],
                [
                    C,
                    -gamma_y * np.eye(y_dim),
                    np.zeros([y_dim, z_dim]),
                ],
                [
                    G,
                    np.zeros([z_dim, y_dim]),
                    -sigma_inv - gamma_z * np.eye(z_dim),
                ],
            ]
        )

        rhs = -np.concatenate([D1L, c(x), g(x) + (mu / z)])

        dxyz = lin_sys_solver(lhs, rhs)

        dx, dy, dz = split_xyz_vars(dxyz)

        ds = -(g(x) + s) + gamma_z * dz - G @ dx

        dxsyz = combine_xsyz_vars(x=dx, s=ds, y=dy, z=dz)
        error = np.linalg.norm(lhs @ dxyz - rhs)
        return dxsyz, error

    def compute_search_direction_symmetric_indirect_method_2x2(
        *, x, s, y, z, mu
    ):
        y_dim = y.shape[0]

        def al_x(xx):
            return barrier_augmented_lagrangian(
                np.concatenate([xx, s, y, z]), mu=mu
            )

        D1L = jax.grad(al_x)(x)
        D2L = project_psd_cone(
            jax.hessian(al_x)(x),
            delta=min_delta,
            use_lapack=psd_use_lapack,
            iterate=psd_iterate,
        )
        C = jax.jacfwd(c)(x)
        G = jax.jacfwd(g)(x)
        sigma = np.diag(z / (s + gamma_z * z))
        lhs = np.block(
            [
                [D2L + G.T @ sigma @ G, C.T],
                [
                    C,
                    -gamma_y * np.eye(y_dim),
                ],
            ]
        )

        rhs = -np.concatenate([D1L + G.T @ sigma @ (g(x) + (mu / z)), c(x)])

        dxy = lin_sys_solver(lhs, rhs)

        dx, dy = split_xy_vars(dxy)

        dz = sigma @ (g(x) + G @ dx + (mu / z))

        ds = -(g(x) + s) + gamma_z * dz - G @ dx

        dxsyz = combine_xsyz_vars(x=dx, s=ds, y=dy, z=dz)
        error = np.linalg.norm(lhs @ dxy - rhs)
        return dxsyz, error

    def compute_search_direction_dispatcher(*, x, s, y, z, mu):
        match lin_sys_formulation:
            case LinearSystemFormulation.AUTODIFF:
                return compute_search_direction_autodiff(
                    x=x, s=s, y=y, z=z, mu=mu
                )
            case LinearSystemFormulation.STABLE_DIRECT_4x4:
                return compute_search_direction_stable_direct_method(
                    x=x, s=s, y=y, z=z, mu=mu
                )
            case LinearSystemFormulation.SYMMETRIC_DIRECT_4x4:
                return compute_search_direction_symmetric_direct_method_4x4(
                    x=x, s=s, y=y, z=z, mu=mu
                )
            case LinearSystemFormulation.SYMMETRIC_INDIRECT_3x3:
                return compute_search_direction_symmetric_indirect_method_3x3(
                    x=x, s=s, y=y, z=z, mu=mu
                )
            case LinearSystemFormulation.SYMMETRIC_INDIRECT_2x2:
                return compute_search_direction_symmetric_indirect_method_2x2(
                    x=x, s=s, y=y, z=z, mu=mu
                )

    def compute_restoration_direction(*, x, s, y, z):
        del y, z
        x_dim = x.shape[0]
        cx = c(x)
        gx = g(x)
        C = jax.jacfwd(c)(x)
        G = jax.jacfwd(g)(x)
        lhs = C.T @ C + G.T @ G + min_delta * np.eye(x_dim)
        rhs = -(C.T @ cx + G.T @ (gx + s))
        dx = lin_sys_solver(lhs, rhs)
        ds = -(gx + s) - G @ dx
        dy = np.zeros_like(ws_y)
        dz = np.zeros_like(ws_z)
        dxsyz = combine_xsyz_vars(x=dx, s=ds, y=dy, z=dz)
        error = np.linalg.norm(lhs @ dx - rhs)
        return dxsyz, error

    def optimization_loop(inputs):
        x = inputs["x"]
        s = inputs["s"]
        y = inputs["y"]
        z = inputs["z"]
        iteration = inputs["iteration"]
        filter_phi_ref = inputs["filter_phi_ref"]
        filter_f_ref = inputs["filter_f_ref"]
        restoration_mode = inputs["restoration_mode"]
        prev_phi = inputs["prev_phi"]
        prev_ineq_inf = inputs["prev_ineq_inf"]
        plateau_count = inputs["plateau_count"]

        mu = np.maximum(adaptive_mu(s=s, z=z, iteration=iteration), mu_min)

        dxsyz, lin_sys_error = compute_search_direction_dispatcher(
            x=x, s=s, y=y, z=z, mu=mu
        )
        dx, ds, dy, dz = split_xsyz_vars(dxsyz)

        tau = np.maximum(tau_min, np.where(mu > 0.0, 1.0 - mu, 0.0))

        # s + alpha_s_max * ds >= (1 - tau) * s
        mod_ds = np.minimum(ds, np.full_like(ds, -1e-12))
        alpha_s_max = np.minimum((-tau * s / mod_ds).min(), 1.0)

        # z + alpha_z_max * dz >= (1 - tau) * z
        mod_dz = np.minimum(dz, np.full_like(dz, -1e-12))
        alpha_z_max = np.minimum((-tau * z / mod_dz).min(), 1.0)

        rho = get_rho(x=x, s=s, dx=dx, ds=ds, mu=mu)
        m = partial(merit_function, mu=mu, rho=rho)
        m_slope = merit_function_slope(x=x, s=s, dx=dx, ds=ds, mu=mu, rho=rho)
        m0 = m(x=x, s=s)
        phi0 = feasibility_measure(x=x, s=s)
        def ls_body(alpha):
            return alpha * line_search_factor

        def merit_accept(alpha):
            return (
                m(
                    x=(x + alpha * dx),
                    s=(s + alpha * ds),
                )
                <= m0 + armijo_factor * m_slope * alpha
            )

        def filter_accept(alpha):
            phi_trial = feasibility_measure(x=(x + alpha * dx), s=(s + alpha * ds))
            f_trial = f(x + alpha * dx)
            return (
                phi_trial <= (1.0 - feasibility_armijo_factor * alpha) * filter_phi_ref
            ) | (
                (f_trial <= filter_f_ref - armijo_factor * filter_phi_ref)
                & (phi_trial <= (1.0 + feasibility_armijo_factor) * filter_phi_ref)
            )

        def ls_continue(alpha):
            return np.logical_and(
                np.logical_not(
                    np.logical_or(merit_accept(alpha), filter_accept(alpha))
                ),
                alpha > line_search_min_step_size,
            )

        def feas_ls_continue(alpha):
            return np.logical_and(
                feasibility_measure(x=(x + alpha * dx), s=(s + alpha * ds))
                > (1.0 - feasibility_armijo_factor * alpha) * filter_phi_ref,
                alpha > line_search_min_step_size,
            )

        alpha_main = jax.lax.cond(
            restoration_mode,
            lambda a0: jax.lax.while_loop(feas_ls_continue, ls_body, a0),
            lambda a0: jax.lax.cond(
                m_slope < 0.0,
                lambda aa0: jax.lax.while_loop(ls_continue, ls_body, aa0),
                lambda aa0: aa0,
                a0,
            ),
            alpha_s_max,
        )
        main_ok = jax.lax.cond(
            restoration_mode,
            lambda _: filter_accept(alpha_main),
            lambda _: np.logical_or(merit_accept(alpha_main), filter_accept(alpha_main)),
            operand=None,
        )

        resto_dxsyz, resto_lin_sys_error = jax.lax.cond(
            np.logical_or(restoration_mode, np.logical_not(main_ok)),
            lambda _: compute_restoration_direction(x=x, s=s, y=y, z=z),
            lambda _: (
                np.zeros_like(dxsyz),
                np.array(0.0, dtype=lin_sys_error.dtype),
            ),
            operand=None,
        )
        resto_dx, resto_ds, resto_dy, resto_dz = split_xsyz_vars(resto_dxsyz)
        mod_resto_ds = np.minimum(resto_ds, np.full_like(resto_ds, -1e-12))
        alpha_s_max_resto = np.minimum((-tau * s / mod_resto_ds).min(), 1.0)

        def resto_ls_continue(alpha):
            return np.logical_and(
                feasibility_measure(
                    x=(x + alpha * resto_dx),
                    s=(s + alpha * resto_ds),
                )
                > (1.0 - feasibility_armijo_factor * alpha) * filter_phi_ref,
                alpha > line_search_min_step_size,
            )

        alpha_resto = jax.lax.while_loop(
            resto_ls_continue,
            ls_body,
            alpha_s_max_resto,
        )
        phi_resto = feasibility_measure(
            x=(x + alpha_resto * resto_dx),
            s=(s + alpha_resto * resto_ds),
        )
        resto_ok = phi_resto < np.minimum(phi0, filter_phi_ref)

        use_main_step = main_ok
        alpha = jax.lax.cond(
            main_ok,
            lambda _: alpha_main,
            lambda _: jax.lax.cond(
                resto_ok,
                lambda __: alpha_resto,
                lambda __: np.array(0.0, dtype=alpha_s_max.dtype),
                operand=None,
            ),
            operand=None,
        )
        step_dx = jax.lax.cond(
            use_main_step,
            lambda _: dx,
            lambda _: resto_dx,
            operand=None,
        )
        step_ds = jax.lax.cond(
            use_main_step,
            lambda _: ds,
            lambda _: resto_ds,
            operand=None,
        )
        step_dy = jax.lax.cond(
            use_main_step,
            lambda _: dy,
            lambda _: resto_dy,
            operand=None,
        )
        step_dz = jax.lax.cond(
            use_main_step,
            lambda _: dz,
            lambda _: resto_dz,
            operand=None,
        )
        step_alpha_z_max = jax.lax.cond(
            use_main_step,
            lambda _: alpha_z_max,
            lambda _: np.array(1.0, dtype=alpha_z_max.dtype),
            operand=None,
        )
        step_lin_sys_error = jax.lax.cond(
            use_main_step,
            lambda _: lin_sys_error,
            lambda _: resto_lin_sys_error,
            operand=None,
        )

        dual_alpha = np.minimum(alpha, step_alpha_z_max)

        new_x = x + alpha * step_dx
        new_s = np.maximum(s + alpha * step_ds, positivity_floor)
        new_y = y + dual_alpha * step_dy
        new_z = np.maximum(z + dual_alpha * step_dz, positivity_floor)
        new_phi = feasibility_measure(x=new_x, s=new_s)
        new_f = f(new_x)
        accepted = alpha > 0.0
        used_resto = np.logical_and(np.logical_not(main_ok), resto_ok)
        stalled_feas = np.logical_and(accepted, new_phi > 0.98 * phi0)
        new_filter_phi_ref = np.where(
            accepted,
            np.minimum(filter_phi_ref, new_phi),
            filter_phi_ref,
        )
        new_filter_f_ref = np.where(
            accepted,
            np.minimum(filter_f_ref, new_f),
            filter_f_ref,
        )
        new_restoration_mode = np.where(
            used_resto,
            True,
            np.where(main_ok, stalled_feas, restoration_mode),
        )

        if print_logs:
            accept_code = np.where(main_ok, 1, np.where(resto_ok, 2, 0))
            jax.debug.print(
                "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10}",
                iteration,
                alpha,
                m(x=new_x, s=new_s),
                f(new_x),
                np.linalg.norm(c(new_x)),
                np.linalg.norm(g(new_x) + new_s),
                m_slope,
                alpha_s_max,
                alpha_z_max,
                np.linalg.norm(step_dx),
                np.linalg.norm(step_ds),
                np.linalg.norm(step_dy),
                np.linalg.norm(step_dz),
                mu,
                step_lin_sys_error,
                accept_code,
            )

        def al_x(xx):
            return barrier_augmented_lagrangian(
                np.concatenate([xx, new_s, new_y, new_z]), mu=mu
            )

        new_grad_al = jax.grad(al_x)(new_x)
        new_c = c(new_x)
        new_g = g(new_x)
        new_merit = m(x=new_x, s=new_s)

        if trace_length > 0:
            accept_code = np.where(main_ok, 1, np.where(resto_ok, 2, 0))
            trace_idx = np.minimum(iteration, trace_length - 1)
            comp_meas = complementarity_measure(s=new_s, z=new_z, mu=mu)
            dual_meas = np.abs(new_grad_al).max()
            trace_alpha = inputs["trace_alpha"].at[trace_idx].set(alpha)
            trace_mu = inputs["trace_mu"].at[trace_idx].set(mu)
            trace_phi = inputs["trace_phi"].at[trace_idx].set(new_phi)
            trace_merit = inputs["trace_merit"].at[trace_idx].set(new_merit)
            trace_eq = inputs["trace_eq"].at[trace_idx].set(np.linalg.norm(new_c))
            trace_ineq = inputs["trace_ineq"].at[trace_idx].set(
                np.linalg.norm(new_g + new_s)
            )
            trace_accept = inputs["trace_accept"].at[trace_idx].set(accept_code)
            trace_linsys = inputs["trace_linsys"].at[trace_idx].set(step_lin_sys_error)
            trace_comp = inputs["trace_comp"].at[trace_idx].set(comp_meas)
            trace_dual = inputs["trace_dual"].at[trace_idx].set(dual_meas)
        else:
            trace_alpha = inputs["trace_alpha"]
            trace_mu = inputs["trace_mu"]
            trace_phi = inputs["trace_phi"]
            trace_merit = inputs["trace_merit"]
            trace_eq = inputs["trace_eq"]
            trace_ineq = inputs["trace_ineq"]
            trace_accept = inputs["trace_accept"]
            trace_linsys = inputs["trace_linsys"]
            trace_comp = inputs["trace_comp"]
            trace_dual = inputs["trace_dual"]

        dual_inf = np.abs(new_grad_al).max()
        primal_eq_inf = np.abs(new_c).max()
        primal_ineq_inf = np.abs(new_g + new_s).max()
        comp_inf = complementarity_measure(s=new_s, z=new_z, mu=mu)
        hard_converged = np.logical_and(
            np.maximum(
                np.maximum(dual_inf, primal_eq_inf),
                np.maximum(primal_ineq_inf, comp_inf),
            )
            < max_kkt_violation,
            mu <= np.maximum(10.0 * mu_min, max_kkt_violation),
        )
        soft_converged = np.logical_and(
            soft_stop_enabled,
            np.logical_and(
                mu <= soft_mu_tol,
                np.logical_and(
                    comp_inf <= soft_comp_tol,
                    np.logical_and(
                        dual_inf <= soft_dual_tol,
                        np.logical_and(
                            primal_eq_inf <= soft_eq_tol,
                            primal_ineq_inf <= soft_ineq_tol,
                        ),
                    ),
                ),
            ),
        )
        phi_improvement = np.maximum(prev_phi - new_phi, 0.0)
        ineq_improvement = np.maximum(prev_ineq_inf - primal_ineq_inf, 0.0)
        plateau_ready = np.logical_and(
            mu <= np.maximum(soft_mu_tol, mu_min),
            np.logical_and(
                comp_inf <= np.maximum(soft_comp_tol, max_kkt_violation),
                dual_inf <= np.maximum(soft_dual_tol, max_kkt_violation),
            ),
        )
        plateau_hit = np.logical_and(
            plateau_stop_enabled,
            np.logical_and(
                plateau_ready,
                np.logical_and(
                    phi_improvement <= plateau_phi_tol,
                    ineq_improvement <= plateau_ineq_tol,
                ),
            ),
        )
        new_plateau_count = np.where(plateau_hit, plateau_count + 1, 0)
        plateau_converged = np.logical_and(
            plateau_stop_enabled,
            np.logical_and(
                iteration + 1 >= plateau_min_iterations,
                new_plateau_count >= plateau_patience,
            ),
        )
        converged = np.logical_or(
            np.logical_or(hard_converged, soft_converged),
            plateau_converged,
        )
        should_continue = np.logical_and(
            np.logical_not(converged), (iteration + 1) < max_iterations
        )

        return {
            "x": new_x,
            "s": new_s,
            "y": new_y,
            "z": new_z,
            "iteration": iteration + 1,
            "should_continue": should_continue,
            "converged": converged,
            "filter_phi_ref": new_filter_phi_ref,
            "filter_f_ref": new_filter_f_ref,
            "restoration_mode": new_restoration_mode,
            "prev_phi": new_phi,
            "prev_ineq_inf": primal_ineq_inf,
            "plateau_count": new_plateau_count,
            "trace_alpha": trace_alpha,
            "trace_mu": trace_mu,
            "trace_phi": trace_phi,
            "trace_merit": trace_merit,
            "trace_eq": trace_eq,
            "trace_ineq": trace_ineq,
            "trace_accept": trace_accept,
            "trace_linsys": trace_linsys,
            "trace_comp": trace_comp,
            "trace_dual": trace_dual,
        }

    def continuation_criteria(inputs):
        return inputs["should_continue"]

    inputs = {
        "x": ws_x,
        "s": ws_s,
        "y": ws_y,
        "z": ws_z,
        "iteration": 0,
        "should_continue": True,
        "converged": False,
        "filter_phi_ref": feasibility_measure(x=ws_x, s=ws_s),
        "filter_f_ref": f(ws_x),
        "restoration_mode": False,
        "prev_phi": feasibility_measure(x=ws_x, s=ws_s),
        "prev_ineq_inf": np.abs(g(ws_x) + ws_s).max(),
        "plateau_count": np.array(0, dtype=np.int32),
        "trace_alpha": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_mu": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_phi": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_merit": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_eq": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_ineq": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_accept": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_linsys": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_comp": np.zeros((trace_length,), dtype=ws_x.dtype),
        "trace_dual": np.zeros((trace_length,), dtype=ws_x.dtype),
    }

    if print_logs:
        jax.debug.print(
            "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(
                "iteration",
                "alpha",
                "merit",
                "f",
                "|c|",
                "|g+s|",
                "m_slope",
                "alpha_s_m",
                "alpha_z_m",
                "|dx|",
                "|ds|",
                "|dy|",
                "|dz|",
                "mu",
                "linsys_res",
                "accept",
            )
        )

    outputs = jax.lax.while_loop(
        continuation_criteria,
        optimization_loop,
        inputs,
    )

    return {
        "x": outputs["x"],
        "s": outputs["s"],
        "y": outputs["y"],
        "z": outputs["z"],
        "iteration": outputs["iteration"],
        "converged": outputs["converged"],
        "trace_alpha": outputs["trace_alpha"],
        "trace_mu": outputs["trace_mu"],
        "trace_phi": outputs["trace_phi"],
        "trace_merit": outputs["trace_merit"],
        "trace_eq": outputs["trace_eq"],
        "trace_ineq": outputs["trace_ineq"],
        "trace_accept": outputs["trace_accept"],
        "trace_linsys": outputs["trace_linsys"],
        "trace_comp": outputs["trace_comp"],
        "trace_dual": outputs["trace_dual"],
    }
