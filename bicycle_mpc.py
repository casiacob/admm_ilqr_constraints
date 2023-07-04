import jax.numpy as jnp
import jax
from admm_iLQR import admm
from jax.config import config
import matplotlib.pyplot as plt
from utils import discretize_dynamics

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")


def project_state_constraints(car_position_consensus: jnp.ndarray) -> jnp.ndarray:
    def ellipse_projection(point_x, point_y, ellipse_a, ellipse_b, cnter_x, center_y):
        theta = jnp.arctan2(point_y - center_y, point_x - cnter_x)
        k = (ellipse_a * ellipse_b) / jnp.sqrt(
            ellipse_b**2 * jnp.cos(theta) ** 2 + ellipse_a**2 * jnp.sin(theta) ** 2
        )
        xbar = k * jnp.cos(theta) + cnter_x
        ybar = k * jnp.sin(theta) + center_y
        return jnp.hstack((xbar, ybar))

    px, py = car_position_consensus

    # ellipse obstacle parameters
    ea = 5.0
    eb = 2.5
    cx = 15.0
    cy = -1.0
    # xc2 = 50.0
    # yc2 = 1.0

    # ellipse constraint
    S = jnp.diag(jnp.array([1.0 / ea**2, 1.0 / eb**2]))
    dxy = jnp.array([px - cx, py - cy])
    violation = 1 - dxy.T @ S @ dxy > 0
    projected_car_position = jnp.where(
        violation, ellipse_projection(px, py, ea, eb, cx, cy), car_position_consensus
    )

    # S = jnp.diag(jnp.array([1.0 / ea**2, 1.0 / eb**2]))
    # dxy = jnp.array([px - xc2, py - yc2])
    # violation = 1 - dxy.T @ S @ dxy > 0
    # zx = jnp.where(violation, ellipse_projection(ea, eb, xc2, yc2), zx)
    return projected_car_position


def project_control_constraints(control_consensus: jnp.ndarray) -> jnp.ndarray:
    acceleration_ub = 1.5
    acceleration_lb = -3
    steering_ub = 0.6
    steering_lb = -0.6
    lb = jnp.array([acceleration_lb, steering_lb])
    ub = jnp.array([acceleration_ub, steering_ub])
    control_consensus = jnp.where(control_consensus <= lb, lb, control_consensus)
    control_consensus = jnp.where(control_consensus >= ub, ub, control_consensus)
    return control_consensus


def projection(
    state: jnp.ndarray,
    control: jnp.ndarray,
    dual: jnp.ndarray,
    penalty: jnp.ndarray,
    A: jnp.ndarray,
) -> jnp.ndarray:
    y = jnp.hstack((state, control))
    z0 = A @ y + 1.0 / penalty * dual
    constrained_states = 2

    # project the xy position onto the ellipse
    zx0 = z0[:constrained_states]
    zx0 = project_state_constraints(zx0)

    # project the controls onto box constraints
    zu0 = z0[constrained_states:]
    zu0 = project_control_constraints(zu0)

    z0 = jnp.hstack((zx0, zu0))
    return z0


def stage_cost(state: jnp.ndarray, control: jnp.ndarray, ref: jnp.ndarray) -> float:
    state_penalty = jnp.diag(jnp.array([0.0, 1.0, 5.0, 0.0]))
    control_penalty = jnp.diag(jnp.array([1.0, 10.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    c = c + control.T @ control_penalty @ control
    return c * 0.5


def final_cost(state: jnp.ndarray, ref: jnp.ndarray) -> float:
    state_penalty = jnp.diag(jnp.array([0.0, 1.0, 5.0, 0.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    return c * 0.5


def car(state: jnp.ndarray, control: jnp.ndarray):
    lf = 1.06
    lr = 1.85
    x, y, v, phi = state
    acceleration, steering = control
    beta = jnp.arctan(jnp.tan(steering * (lr / (lf + lr))))
    return jnp.hstack(
        (
            v * jnp.cos(phi + beta),
            v * jnp.sin(phi + beta),
            acceleration,
            v / lr * jnp.sin(beta),
        )
    )


simulation_step = 0.1
downsampling = 1
dynamics = discretize_dynamics(
    ode=car, simulation_step=simulation_step, downsampling=downsampling
)


def plot_traj(states, controls):
    cx1 = 15.0
    cy1 = -1.0
    # cx2 = 50.0
    # cy2 = 1.0
    a = 5.0  # radius on the x-axis
    b = 2.5  # radius on the y-axis
    t = jnp.linspace(0, 2 * jnp.pi, 150)
    plt.plot(cx1 + a * jnp.cos(t), cy1 + b * jnp.sin(t), color="red")
    # plt.plot(cx2 + a * jnp.cos(t), cy2 + b * jnp.sin(t), color="red")
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.ylim([-10, 10])
    plt.show()
    plt.plot(states[:, 2])
    plt.ylabel("velocity")
    plt.show()
    plt.plot(states[:, 3])
    plt.ylabel("yaw")
    plt.show()
    plt.plot(controls[:, 0])
    plt.ylabel("acceleration")
    plt.show()
    plt.plot(controls[:, 1])
    plt.ylabel("steering")
    plt.show()


cons_vars = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

horizon = 10
penalty_param = 10.0
mean = jnp.array([0.0, 0.0])
sigma = jnp.array([0.01, 0.01])
key = jax.random.PRNGKey(1325)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 2))
z = jnp.zeros((horizon, 4))
l = jnp.zeros((horizon, 4))
x0 = jnp.array([0.0, 0.0, 5.0, 0.0])
xd = jnp.array([0.0, 0.0, 8.0, 0.0])


def mpc_body(carry, inp):
    jax.debug.print(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    prev_state, prev_control, prev_consensus, prev_dual = carry
    control, consensus, dual, it = admm(
        prev_state,
        prev_control,
        xd,
        prev_consensus,
        prev_dual,
        cons_vars,
        stage_cost,
        final_cost,
        dynamics,
        projection,
        penalty_param,
    )
    next_state = dynamics(prev_state, control[0])
    return (next_state, control, consensus, dual), (next_state, control[0], it)


_, mpc_out = jax.lax.scan(mpc_body, (x0, u, z, l), None, length=60)
mpc_states, mpc_controls, iterations = mpc_out
mpc_states = jnp.vstack((x0, mpc_states))
plot_traj(mpc_states, mpc_controls)