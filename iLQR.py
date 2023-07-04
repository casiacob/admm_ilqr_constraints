import jax
import jax.numpy as jnp
from backward_pass import bwd_pass
from forward_pass import fwd_pass


def ilqr(
    nominal_states0: jnp.ndarray,
    nominal_controls0: jnp.ndarray,
    goal_state: jnp.ndarray,
    consensus_vars: jnp.ndarray,
    lagrange_mult: jnp.ndarray,
    penalty_param: float,
    stage_cost,
    final_cost,
    constrained_variables_selector,
    dynamics,
):
    def stage_augmented_lagrangian(x, u, xd, z, l, sigma, A):
        y = jnp.hstack((x, u))
        aug_term = (
            sigma
            / 2.0
            * (A @ y - z + 1.0 / sigma * l).T
            @ (A @ y - z + 1.0 / sigma * l)
        )
        L_sigma = stage_cost(x, u, xd) + aug_term
        return L_sigma

    def total_augmented_cost(x, u, xd, z, l, sigma, A):
        L_k = jax.vmap(
            stage_augmented_lagrangian, in_axes=(0, 0, None, 0, 0, None, None)
        )(x[:-1], u, xd, z, l, sigma, A)
        J_f = final_cost(x[-1], xd)
        total = J_f + jnp.sum(L_k)
        return total

    def iLQR_routine(val):
        states, controls, reg_param, reg_param_mult_fact, _, loop_counter = val

        # compute initial cost
        cost = total_augmented_cost(
            states,
            controls,
            goal_state,
            consensus_vars,
            lagrange_mult,
            penalty_param,
            constrained_variables_selector,
        )

        # backward pass on nominal states and control trajectory
        ffgain, gain, diff_cost, feasible_bp = bwd_pass(
            states,
            controls,
            goal_state,
            consensus_vars,
            lagrange_mult,
            penalty_param,
            reg_param,
            dynamics,
            final_cost,
            stage_augmented_lagrangian,
            constrained_variables_selector,
        )

        # forward pass on nominal states and control trajectory with gains computed at backward pass
        new_states, new_controls = fwd_pass(states, controls, ffgain, gain, dynamics)

        new_cost = total_augmented_cost(
            new_states,
            new_controls,
            goal_state,
            consensus_vars,
            lagrange_mult,
            penalty_param,
            constrained_variables_selector,
        )

        value_change = cost - new_cost
        gain_ratio = value_change / (-diff_cost)

        def accept_step():
            return (
                reg_param * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                2.0,
                new_states,
                new_controls,
            )

        def reject_step():
            return (
                reg_param * reg_param_mult_fact,
                reg_param_mult_fact * 2.0,
                states,
                controls,
            )

        reg_param, reg_param_mult_fact, states, controls = jax.lax.cond(
            jnp.logical_and(gain_ratio > 0.0, feasible_bp),
            accept_step,
            reject_step,
        )
        # jax.debug.print('reg param {x}', x=reg_param)
        loop_counter += 1

        return (
            states,
            controls,
            reg_param,
            reg_param_mult_fact,
            value_change,
            loop_counter,
        )

    def iLQR_convergence(val):
        _, _, _, _, value_change, loop_counter = val
        exit_condition = jnp.logical_or(
            jnp.abs(value_change) < 1e-16, loop_counter > max_iter
        )
        return jnp.logical_not(exit_condition)

    max_iter = jnp.inf
    reg_param0 = 1.
    reg_param_mult_fact0 = 2.0

    optimal_states, optimal_controls, _, _, _, iterations = jax.lax.while_loop(
        iLQR_convergence,
        iLQR_routine,
        (
            nominal_states0,
            nominal_controls0,
            reg_param0,
            reg_param_mult_fact0,
            jnp.inf,
            0,
        ),
    )
    # jax.debug.print("iLQR converged in {x}", x=iterations)
    return optimal_states, optimal_controls
