import jax.numpy as jnp
from utils import wrap_angle, discretize_dynamics
import jax
from admm_iLQR import admm
import matplotlib.pyplot as plt
from jax.config import config

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")

def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = (
        0.5
        * (_wrapped - goal_state).T
        @ final_state_cost
        @ (_wrapped - goal_state)
    )
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, goal_state: jnp.ndarray
) -> float:

    state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c

def cartpole(
    state: jnp.ndarray, action: jnp.ndarray
) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = state

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
        action
        + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )


simulation_step = 0.05
downsampling = 1
dynamics = discretize_dynamics(
    ode=cartpole, simulation_step=simulation_step, downsampling=downsampling
)


def project_control_constraints(consensus: jnp.ndarray) -> jnp.ndarray:
    acceleration_ub = 50
    acceleration_lb = -50
    consensus = jnp.where(consensus <= acceleration_lb, acceleration_lb, consensus)
    consensus = jnp.where(consensus >= acceleration_ub, acceleration_ub, consensus)
    return consensus


def projection(
    states: jnp.ndarray,
    controls: jnp.ndarray,
    lagrange_mult: jnp.ndarray,
    penalty: float,
    A: jnp.ndarray,
) -> jnp.ndarray:
    y = jnp.hstack((states, controls))
    z0 = A @ y + 1.0 / penalty * lagrange_mult
    z0 = project_control_constraints(z0)
    return z0

def plot_traj(states, controls, conv_iterations):
    plt.plot(states[:, 0], label="cart position")
    plt.plot(states[:, 1], label="angle position")
    plt.legend()
    plt.show()
    plt.plot(states[:, 2], label="cart velocity")
    plt.plot(states[:, 3], label="angle velocity")
    plt.legend()
    plt.show()
    plt.plot(controls)
    plt.show()
    plt.plot(conv_iterations)
    plt.ylabel("admm iterations")
    plt.show()


horizon = 20
A = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0]])  # select constrained states and control A [x' u']'
mean = jnp.array([0.0])
sigma = jnp.array([0.1])
key = jax.random.PRNGKey(1)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 1))
z = jnp.zeros((horizon, 1))
l = jnp.zeros((horizon, 1))
x0 = jnp.array([0.01, -0.01, 0.01, -0.01])
xd = jnp.array([0.0, jnp.pi, 0.0, 0.0])
penalty_param = 20.0


def mpc_body(carry, inp):
    jax.debug.print(
        "mpc iteration {x}", x=inp
    )
    prev_state, prev_control, prev_consensus, prev_dual = carry
    control, consensus, dual, conv_it = admm(
        prev_state,
        prev_control,
        xd,
        prev_consensus,
        prev_dual,
        A,
        transient_cost,
        final_cost,
        dynamics,
        projection,
        penalty_param,
    )
    next_state = dynamics(prev_state, control[0])
    # jax.debug.callback(plot_traj, control)
    # jax.debug.breakpoint()
    return (next_state, control, consensus, dual), (next_state, control[0], conv_it)


_, mpc_out = jax.lax.scan(mpc_body, (x0, u, z, l), jnp.arange(0, 80, 1), length=80)
mpc_states, mpc_controls, iterations = mpc_out
plot_traj(mpc_states, mpc_controls, iterations)
