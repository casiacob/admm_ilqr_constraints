import jax.numpy as jnp
import jax
from admm_iLQR import admm
from jax.config import config
import matplotlib.pyplot as plt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")


#


def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    m = 1412
    lf = 1.06
    lr = 1.85
    kf = -128916
    kr = -85944
    Ts = 0.05
    L = lf * kf - lr * kr
    Iz = 1536.7

    x_pos, y_pos, yaw, x_vel, y_vel, ang_vel = state
    acc, steering = control

    px_next = x_pos + Ts * (x_vel * jnp.cos(yaw) - y_vel * jnp.sin(yaw))
    py_next = y_pos + Ts * (y_vel * jnp.cos(yaw) + x_vel * jnp.sin(yaw))
    phi_next = yaw + Ts * ang_vel
    vx_next = x_vel + Ts * acc
    vy_next = (
        m * x_vel * y_vel
        + Ts * L * ang_vel
        - Ts * kf * steering * x_vel
        - Ts * m * x_vel**2 * ang_vel
    ) / (m * x_vel - Ts * (kf + kr))
    w_next = (
        Iz * x_vel * ang_vel + Ts * L * y_vel - Ts * lf * kf * steering * x_vel
    ) / (Iz * x_vel - Ts * (lf**2 * kf + lr**2 * kr))
    x_next = jnp.hstack((px_next, py_next, phi_next, vx_next, vy_next, w_next))
    return x_next


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
    steering_ub = jnp.pi / 3.0
    steering_lb = -jnp.pi / 3.0
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
    state_penalty = jnp.diag(jnp.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0]))
    control_penalty = jnp.diag(jnp.array([1.0, 1.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    c = c + control.T @ control_penalty @ control
    return c


def final_cost(state: jnp.ndarray, ref: jnp.ndarray) -> float:
    state_penalty = jnp.diag(jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    return c


def plot_traj(states, controls):
    cx1 = 15.0
    cy1 = -1.0
    cx2 = 50.0
    cy2 = 1.0
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
    plt.ylabel("yaw angle")
    plt.show()
    plt.plot(states[:, 3])
    plt.ylabel("x velocity")
    plt.show()
    plt.plot(states[:, 4])
    plt.ylabel("y velocity")
    plt.show()
    plt.plot(states[:, 5])
    plt.ylabel("yaw rate")
    plt.show()
    plt.plot(controls[:, 0])
    plt.ylabel("acceleration")
    plt.show()
    plt.plot(controls[:, 1])
    plt.ylabel("steering")
    plt.show()


cons_vars = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

horizon = 20
penalty_param = 5.0
mean = jnp.array([0.0, 0.0])
sigma = jnp.array([0.01, 0.01])
key = jax.random.PRNGKey(1)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 2))
z = jnp.zeros((horizon, 4))
l = jnp.zeros((horizon, 4))
x0 = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
xd = jnp.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0])

# u, z, l = admm(x0, u, z, l, A, stage_cost, final_cost, dynamics, projection)


def mpc_body(carry, inp):
    jax.debug.print(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    prev_state, prev_control, prev_consensus, prev_dual = carry
    control, consensus, dual, _ = admm(
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

    # def state_traj(x_0, u_inp):
    #     def body_scan(x_prev, act):
    #         x_next = dynamics(x_prev, act)
    #         return x_next, x_next
    #
    #     _, x = jax.lax.scan(body_scan, x_0, u_inp)
    #     x = jnp.vstack((x_0, x))
    #     return x
    # states = state_traj(prev_state, control)
    # jax.debug.callback(plot_traj, states, control)
    # jax.debug.breakpoint()
    return (next_state, control, consensus, dual), (next_state, control[0])


_, mpc_out = jax.lax.scan(mpc_body, (x0, u, z, l), None, length=120)
mpc_states, mpc_controls = mpc_out
plot_traj(mpc_states, mpc_controls)
