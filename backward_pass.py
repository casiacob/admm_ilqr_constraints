import jax.numpy as jnp
import jax.scipy as jcp
from jax.lax import scan, cond
from jax import jacrev, jacfwd, grad, hessian

# from inverted_pendulum import f, stage_augmented_lagrangian, final_cost


def bwd_pass(
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    goal_state: jnp.ndarray,
    consensus_vars: jnp.ndarray,
    lagrange_mult: jnp.ndarray,
    penalty_param: float,
    reg: float,
    dynamics,
    final_cost,
    stage_augmented_lagrangian,
    constrained_variables_selector,
):
    def bwd_step(carry, inp):
        # unpack carry
        Vx, Vxx, convex = carry

        # state, controls, consensus, dual
        state, control, consensus, dual = inp

        # derivatives
        lx = grad(stage_augmented_lagrangian, 0)(
            state,
            control,
            goal_state,
            consensus,
            dual,
            penalty_param,
            constrained_variables_selector,
        )
        fx = jacrev(dynamics, 0)(state, control)
        lu = grad(stage_augmented_lagrangian, 1)(
            state,
            control,
            goal_state,
            consensus,
            dual,
            penalty_param,
            constrained_variables_selector,
        )
        fu = jacrev(dynamics, 1)(state, control)
        lxx = hessian(stage_augmented_lagrangian, 0)(
            state,
            control,
            goal_state,
            consensus,
            dual,
            penalty_param,
            constrained_variables_selector,
        )
        luu = hessian(stage_augmented_lagrangian, 1)(
            state,
            control,
            goal_state,
            consensus,
            dual,
            penalty_param,
            constrained_variables_selector,
        )
        lux = jacfwd(jacrev(stage_augmented_lagrangian, 1), 0)(
            state,
            control,
            goal_state,
            consensus,
            dual,
            penalty_param,
            constrained_variables_selector,
        )

        Qx = lx + fx.T @ Vx
        Qu = lu + fu.T @ Vx
        Qxx = lxx + fx.T @ Vxx @ fx
        Qux = lux + fu.T @ Vxx @ fx
        Quu = luu + fu.T @ Vxx @ fu
        Quu = Quu + reg * jnp.eye(control.shape[0])
        Quu = (Quu + Quu.T) / 2
        convex = jnp.logical_and(convex, jnp.all(jnp.linalg.eigvals(Quu) > 0.0))

        def is_convex():
            chol_and_lower = jcp.linalg.cho_factor(Quu)
            alpha = -jcp.linalg.cho_solve(chol_and_lower, Qu)
            beta = -jcp.linalg.cho_solve(chol_and_lower, Qux)
            return beta, alpha

        def is_indef():
            return jnp.zeros((control.shape[0], state.shape[0])), jnp.zeros_like(
                control
            )

        K, k = cond(convex, is_convex, is_indef)
        Vx = Qx - K.T @ Quu @ k
        Vxx = Qxx - K.T @ Quu @ K
        dV = k.T @ Qu + 0.5 * k.T @ Quu @ k
        return (Vx, Vxx, convex), (k, K, dV)

    xN = nominal_states[-1]
    VxN = grad(final_cost, 0)(xN, goal_state)
    VxxN = hessian(final_cost, 0)(xN, goal_state)

    feasible = jnp.bool_(1.0)

    # loop the backwards step function and compute gains
    carry_out, bwd_pass_out = scan(
        bwd_step,
        (VxN, VxxN, feasible),
        (nominal_states[:-1], nominal_controls, consensus_vars, lagrange_mult),
        reverse=True,
    )

    # return the gain and feed forward gain
    _, _, feasible = carry_out
    ffgain, gain, diff_cost = bwd_pass_out
    diff_cost = jnp.sum(diff_cost)
    return ffgain, gain, diff_cost, feasible
