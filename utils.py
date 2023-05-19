from typing import Callable
import jax.numpy as jnp
from jax.lax import fori_loop


def wrap_angle(x: float) -> float:
    # wrap angle between [0, 2*pi]
    return x % (2.0 * jnp.pi)


def runge_kutta(
    state: jnp.ndarray,
    action: jnp.ndarray,
    ode: Callable,
    step: float,
):
    k1 = ode(state, action)
    k2 = ode(state + 0.5 * step * k1, action)
    k3 = ode(state + 0.5 * step * k2, action)
    k4 = ode(state + step * k3, action)
    return state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def discretize_dynamics(ode: Callable, simulation_step: float, downsampling: int):
    def dynamics(
        state: jnp.ndarray,
        action: jnp.ndarray,
    ):
        def _step(t, state_t):
            next_state = runge_kutta(
                state_t,
                action,
                ode,
                simulation_step,
            )
            return next_state

        return fori_loop(
            lower=0,
            upper=downsampling,
            body_fun=_step,
            init_val=state,
        )

    return dynamics
