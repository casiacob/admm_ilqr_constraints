import jax.numpy as jnp
import jax
from admm_iLQR import admm
import matplotlib.pyplot as plt

constrained_variables_selector = jnp.array([[0.0, 0.0, 1.0]])


def dynamics(state: jnp.ndarray, torque: jnp.ndarray):
    h = 0.05
    angle, angular_vel = state
    angle_next = angle + h * angular_vel
    angular_vel_next = angular_vel + h * jnp.sin(angle) + h * torque
    state_next = jnp.hstack((angle_next, angular_vel_next))
    return state_next


def state_traj(initial_state: jnp.ndarray, torque: jnp.ndarray):
    def body_scan(prev_state, control):
        next_state = dynamics(prev_state, control)
        return next_state, next_state

    _, states = jax.lax.scan(body_scan, initial_state, torque)
    states = jnp.vstack((initial_state, states))
    return states


def project_control_constraints(consensus: jnp.ndarray):
    torque_ub = 0.25
    torque_lb = -0.25
    consensus = jnp.where(consensus <= torque_lb, torque_lb, consensus)
    consensus = jnp.where(consensus >= torque_ub, torque_ub, consensus)
    return consensus


def projection(
    states: jnp.ndarray,
    torques: jnp.ndarray,
    lagrange_mult: jnp.ndarray,
    penalty: float,
    A: jnp.ndarray,
):
    y = jnp.hstack((states, torques))
    z0 = A @ y + 1.0 / penalty * lagrange_mult
    z0 = project_control_constraints(z0)
    return z0


def stage_cost(state: jnp.ndarray, torque: jnp.ndarray):
    Q = jnp.diag(jnp.array([0.005, 0.005, 0.1]))
    # Q = jnp.diag(jnp.array([1., 1., 0.0025]))
    ref = jnp.array([0.0, 0.0, 0.0])
    w = jnp.hstack((state, torque))
    J = (ref - w).T @ Q @ (ref - w)
    return J


def final_cost(final_state: jnp.ndarray):
    Q = jnp.diag(jnp.array([5.0, 5.0]))
    ref = jnp.array([0.0, 0.0])
    J = (ref - final_state).T @ Q @ (ref - final_state)
    return J


def plot_traj(states, controls):
    plt.plot(states[:, 0], label="angle")
    plt.plot(states[:, 1], label="angular velocity")
    plt.legend()
    plt.show()
    plt.plot(controls)
    plt.ylabel("torque")
    plt.show()


N = 100
mean = jnp.array([0.0])
sigma = jnp.array([0.01])
key = jax.random.PRNGKey(1325)
u = mean + sigma * jax.random.normal(key, shape=(N, 1))
z = jnp.zeros((N, 1))
l = jnp.zeros((N, 1))
x0 = jnp.array([-2 * jnp.pi / 3.0, 0.0])
penalty_param = 10.0


def mpc_body(carry, inp):
    jax.debug.print(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    prev_state, prev_control, prev_consensus, prev_dual = carry
    control, consensus, dual = admm(
        prev_state,
        prev_control,
        prev_consensus,
        prev_dual,
        constrained_variables_selector,
        stage_cost,
        final_cost,
        dynamics,
        projection,
        penalty_param,
    )
    next_state = dynamics(prev_state, control[0])
    # jax.debug.callback(plot_traj, control)
    # jax.debug.breakpoint()
    return (next_state, control, consensus, dual), (next_state, control[0])


_, mpc_out = jax.lax.scan(mpc_body, (x0, u, z, l), None, length=400)
mpc_states, mpc_controls = mpc_out
# plot_traj(mpc_controls)
plot_traj(mpc_states, mpc_controls)
