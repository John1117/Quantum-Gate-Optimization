import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from simulator import SinPulseQuantGateSimulator
from optimizers.bayesian_optimizer import BayesianOptimizer
from optimizers.nelder_mead_optimizer import NelderMeadOptimizer


# sim
y_opt_threshold = 1e-15
x_gate = np.array([[0, 1], [1, 0]], np.complex128)
h_gate = np.array([[1, 1], [1, -1]], np.complex128)*2**-0.5
sim = SinPulseQuantGateSimulator(
    target_gate=h_gate,
    gate_time=2**-0.5*1,
    z_const=1,
    z_noise_sigma=0,
    sin_freqs=np.array([1, 3]),
    sin_amps=None,
    sin_amps_bounds=np.array([[-10, 10]], np.float64),
    rk45_rtol=1e-7,
    rk45_atol=1e-8,
    grid_sampling_sigmas=np.array([0], np.float64),
    dtype=np.float64
)

# x0
x0 = np.random.uniform(
    sim.sin_amps_bounds[:, 0],
    sim.sin_amps_bounds[:, 1],
    (3, 2)
)

# bo
bo = BayesianOptimizer(
    objective_fun=sim.to_objective_fun,
    bounds=sim.sin_amps_bounds,
    max_fun_evals=np.inf,
    num_initial_points=1,
    x0=x0,
    y0=None,
    const_value=1,
    const_value_bounds=(1e-1, 1e1),
    matern_len=1,
    matern_len_bounds=(1e-1, 1e1),
    matern_nu=1.5,
    sin_len=None,
    sin_len_bounds=(1e-1, 1e1),
    sin_period=1.0,
    sin_period_bounds=(1e-1, 1e1),
    gpr_optimizer='fmin_l_bfgs_b',
    gpr_num_trials=1,
    acq_fun='EI',  # LCB/EI/PI
    acq_optimizer='Nelder-Mead',  # sampling/L-BFGS-B/Nelder-Mead
    acq_optimizer_options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-6},  # {'xatol': 1e-3, 'fatol': 1e-6}
    acq_num_trials=5,
    random_state=np.random.randint(1000),
    acq_kappa=1.96,
    acq_xi=0.01,
    noise=1e-10,
    y_opt_threshold=y_opt_threshold
)

# nm
nm = NelderMeadOptimizer(
        objective_fun=sim.to_objective_fun,
        bounds=sim.sin_amps_bounds,
        x0=x0,
        y0=None,
        max_fun_evals=np.inf,
        coef_reflect=1,
        coef_expand=2,
        coef_contract=0.5,
        coef_shrink=0.5,
        x_tol=1e-7,
        y_tol=1e-15,
        adapt=False,
        y_opt_threshold=y_opt_threshold
)

# obj
num_x_1d = 50
x_1d = np.linspace(bo.bounds[0, 0], bo.bounds[0, 1], num=num_x_1d)
x1, x2 = np.meshgrid(x_1d, x_1d)
x = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=-1)
y_obj = sim.to_objective_fun(x).reshape(num_x_1d, num_x_1d)
y_sur, y_sigma = bo.surrogate_fun(x)
y_sur = y_sur.reshape(num_x_1d, num_x_1d)

# fig
matplotlib.use('TkAgg')
plt.ion()
fig = plt.figure(num='topo_2d_test', figsize=(15, 15), tight_layout=True)
fig.suptitle('Evals.: {}, BO: {:.3e}, NM: {:.3e}'.format(3, bo.y_opt, nm.y_opt), fontsize=20)
ax = fig.add_subplot(projection='3d')

num_evals = 150
for idx_evals in range(4, num_evals+1):
    bo.run(1)
    nm.run(1)
    y_sur, y_sigma = bo.surrogate_fun(x)
    y_sur = y_sur.reshape(num_x_1d, num_x_1d)

    ax.clear()
    ax.set_xlabel('Pulse Param. #1', fontsize=20, labelpad=20)
    ax.set_ylabel('Pulse Param. #2', fontsize=20, labelpad=20)
    ax.set_zlabel('Avg. Infidelity', fontsize=20, labelpad=20)
    ax.set_xlim(bo.bounds[0, 0] - 1, bo.bounds[0, 1] + 1)
    ax.set_ylim(bo.bounds[1, 0] - 1, bo.bounds[1, 1] + 1)
    ax.set_zlim(-0.1, 1.1)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_zscale('linear')
    ax.grid(True)

    cf = ax.contourf(x1, x2, abs(y_sur - y_obj), zdir='z', offset=-0.1, cmap='gray', vmin=0, vmax=0.5)

    ax.plot_surface(x1, x2, y_obj, color=(0, 0, 0, 0.1), edgecolor=(0, 0, 0, 0.2), rcount=num_x_1d//2, lw=0.5)
    ax.plot_surface(x1, x2, y_sur, color=(0, 0, 1, 0.1), edgecolor=(0, 0, 1, 0.2), rcount=num_x_1d//2, lw=0.5, alpha=0.1)
    ax.scatter(bo.x_evals[:, 0], bo.x_evals[:, 1], bo.y_evals, c='b', s=50)
    ax.scatter(bo.x_opt[0], bo.x_opt[1], bo.y_opt, c='b', marker='*', s=500)

    ax.scatter(nm.x_evals[:, 0], nm.x_evals[:, 1], nm.y_evals, c='r', s=50)
    ax.scatter(nm.x_opt[0], nm.x_opt[1], nm.y_opt, c='r', marker='*', s=500)

    fig.suptitle('Evals.: {}, BO: {:.3e}, NM: {:.3e}'.format(idx_evals, bo.y_opt, nm.y_opt), fontsize=20)
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()
