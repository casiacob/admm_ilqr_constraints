import jax.numpy as jnp
from utils import discretize_dynamics, wrap_angle
import jax
import matplotlib.pyplot as plt
from admm_iLQR import admm
from jax.config import config

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([2e0, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state

    c = 0.5 * _wrapped.T @ final_state_cost @ _wrapped
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, goal_state: jnp.ndarray
) -> float:
    state_cost = jnp.diag(jnp.array([2e0, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state

    c = 0.5 * _wrapped.T @ state_cost @ _wrapped
    c += 0.5 * action.T @ action_cost @ action
    return c


def pendulum(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    gravity = 9.81
    length = 1.0
    mass = 1.0
    damping = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            -gravity / length * jnp.sin(position)
            + (action - damping * velocity) / (mass * length**2),
        )
    )


simulation_step = 0.01
downsampling = 5
dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=simulation_step, downsampling=downsampling
)


def project_control_constraints(consensus: jnp.ndarray) -> jnp.ndarray:
    torque_ub = 5.0
    torque_lb = -5.0
    consensus = jnp.where(consensus <= torque_lb, torque_lb, consensus)
    consensus = jnp.where(consensus >= torque_ub, torque_ub, consensus)
    return consensus


def projection(
    states: jnp.ndarray,
    torques: jnp.ndarray,
    lagrange_mult: jnp.ndarray,
    penalty: float,
    A: jnp.ndarray,
) -> jnp.ndarray:
    y = jnp.hstack((states, torques))
    z0 = A @ y + 1.0 / penalty * lagrange_mult
    z0 = project_control_constraints(z0)
    return z0


def plot_traj(states, controls, conv_iterations):
    plt.plot(states[:, 0], label="angle")
    plt.plot(states[:, 1], label="angular velocity")
    plt.legend()
    plt.show()
    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.plot(controls)
    # ax1.set_ylabel("torque")
    # ax2.plot(conv_iterations)
    # ax2.set_ylabel("admm iterations")
    # plt.show()
    plt.plot(controls)
    plt.show()


horizon = 40
A = jnp.array([[0.0, 0.0, 1.0]])  # select constrained states and control A [x' u']'
mean = jnp.array([0.0])
sigma = jnp.array([1.0])
key = jax.random.PRNGKey(1325)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 1))
z = jnp.zeros((horizon, 1))
l = jnp.zeros((horizon, 1))
x0 = jnp.array([wrap_angle(0.01), -0.01])
xd = jnp.array((jnp.pi, 0.0))
penalty_param = 4.0


def mpc_body(carry, inp):
    # jax.debug.print(
    #     "mpc iteration {x}", x = inp
    # )
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
    return (next_state, control, consensus, dual), (next_state, control[0], conv_it)


_, mpc_out = jax.lax.scan(mpc_body, (x0, u, z, l), jnp.arange(0, 100, 1), length=100)
mpc_states, mpc_controls, iterations = mpc_out
mpc_states = jnp.vstack((x0, mpc_states))
plot_traj(-mpc_states, -mpc_controls, iterations)
