# from inverted_pendulum import f
import jax.numpy as jnp
import jax


def fwd_pass(
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    ffgain: jnp.ndarray,
    gain: jnp.ndarray,
    dynamics,
):
    def fwd_step(prev_state, inp):
        state, control, control_ff_gain, control_gain = inp
        control = control + control_ff_gain + control_gain @ (prev_state - state)
        next_state = dynamics(prev_state, control)
        return next_state, (next_state, control)

    _, new_trajectory = jax.lax.scan(
        fwd_step,
        nominal_states[0],
        (nominal_states[:-1], nominal_controls, ffgain, gain),
    )
    new_states, new_controls = new_trajectory
    new_states = jnp.vstack([nominal_states[0], new_states])
    return new_states, new_controls
