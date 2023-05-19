from typing import Callable
import jax.numpy as jnp
from jax.lax import while_loop, scan
from jax import vmap, debug
from iLQR import ilqr


def admm(
    initial_state: jnp.ndarray,
    initial_controls: jnp.ndarray,
    goal_state: jnp.ndarray,
    consensus_vars_init: jnp.ndarray,
    lagrange_mult_init: jnp.ndarray,
    constrained_variables_selector: jnp.ndarray,
    stage_cost: Callable,
    final_cost: Callable,
    dynamics: Callable,
    projection: Callable,
    penalty_param0: float,
):
    def state_traj(x0, u):
        def body_scan(x_prev, control):
            x_next = dynamics(x_prev, control)
            return x_next, x_next

        _, x = scan(body_scan, x0, u)
        x = jnp.vstack((x0, x))
        return x

    def dual_ascent(x, u, z, l, sigma):
        y = jnp.hstack((x, u))
        l = l + sigma * (constrained_variables_selector @ y - z)
        return l

    def primal_residual(x, u, z):
        y = jnp.hstack((x, u))
        return constrained_variables_selector @ y - z

    def admm_iteration(val):
        controls, consensus, lagrange_mult, loop_cnt, _, _, penalty_param = val
        consensus_prev = consensus

        # ADMM step 1: ilqr
        states = state_traj(initial_state, controls)
        states, controls = ilqr(
            states,
            controls,
            goal_state,
            consensus,
            lagrange_mult,
            penalty_param,
            stage_cost,
            final_cost,
            constrained_variables_selector,
            dynamics,
        )

        # ADMM step 2: projection
        consensus = vmap(projection, in_axes=(0, 0, 0, None, None))(
            states[:-1],
            controls,
            lagrange_mult,
            penalty_param,
            constrained_variables_selector,
        )

        # ADMM step 3: dual ascent
        lagrange_mult = vmap(dual_ascent, in_axes=(0, 0, 0, 0, None))(
            states[:-1], controls, consensus, lagrange_mult, penalty_param
        )

        # primal and dual residuals' infinity norms
        rp_infty = jnp.max(
            jnp.abs(vmap(primal_residual)(states[:-1], controls, consensus))
        )

        # adaptive penalty param update
        rd_infty = jnp.max(jnp.abs(consensus - consensus_prev))
        # res_ratio = jnp.sqrt(rp_infty / rd_infty)
        # res_ratio = 1.
        # res_ratio = jnp.where(res_ratio < jnp.inf, res_ratio, 1.0)
        # penalty_param = penalty_param * jnp.where(res_ratio >= 5, res_ratio, 1.0)
        loop_cnt += 1
        return (
            controls,
            consensus,
            lagrange_mult,
            loop_cnt,
            rp_infty,
            rd_infty,
            penalty_param,
        )

    def admm_conv(val):
        _, _, _, loop_cnt, rp_infty, rd_infty, penalty_param = val
        # debug.print('penalty_param {x}', x=penalty_param)
        exit_condition = jnp.logical_and(rp_infty < 1e-4, rd_infty < 1e-3)
        return jnp.logical_and(loop_cnt < max_iter, jnp.logical_not(exit_condition))

    max_iter = jnp.inf
    (
        optimal_controls,
        optimal_consensus,
        optimal_lagrange,
        iterations,
        max_primal_residual,
        max_dual_residual,
        _,
    ) = while_loop(
        admm_conv,
        admm_iteration,
        (
            initial_controls,
            consensus_vars_init,
            lagrange_mult_init,
            0,
            jnp.inf,
            jnp.inf,
            penalty_param0,
        ),
    )
    debug.print("ADMM converged in {x}", x=iterations)
    debug.print("||rp||_infty = {x}", x=max_primal_residual)
    debug.print("||rd||_infty = {x}", x=max_dual_residual)
    return optimal_controls, optimal_consensus, optimal_lagrange, iterations
