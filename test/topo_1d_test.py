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
    sin_freqs=np.array([1]),
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
    (2, 1)
)

# bo
bo = BayesianOptimizer(
    objective_fun=sim.to_objective_fun,  # lambda t: np.log10(sim.to_objective_fun(t)),
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
    gpr_num_trials=5,
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
num_x = 500
x = np.linspace(sim.sin_amps_bounds[0, 0], sim.sin_amps_bounds[0, 1], num=num_x)
y_obj = sim.to_objective_fun(x.reshape(-1, 1)).reshape(-1)

y_sur, y_sigma = bo.surrogate_fun(x.reshape(-1, 1))
y_sur = y_sur.reshape(-1)

# fig
matplotlib.use('TkAgg')
plt.ion()
fig = plt.figure(num='topo_1d_test', figsize=(20, 8), tight_layout=True)
fig.suptitle('Evals.: {}, BO: {:.2e}, NM: {:.2e}'.format(2, bo.y_opt, nm.y_opt), fontsize=20)
gs = fig.add_gridspec(nrows=2, ncols=1)

# ax1
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_ylabel('Acq. Score', fontsize=20)
ax1.tick_params(axis='both', labelsize=20)
ax1.set_xscale('linear')
ax1.set_yscale('linear')
ax1.grid(True)

l_acq, = ax1.plot(x, bo.acquisition_fun(x.reshape(-1, 1)).reshape(-1), 'g-', lw=2.5)
vl1_acq = ax1.axvline(bo.x_evals[-1], ls='--', c='k', lw=2.5)

# ax2
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlabel('Pulse Params.', fontsize=20)
ax2.set_ylabel('Avg. Infidelity', fontsize=20)
ax2.tick_params(axis='both', labelsize=20)
ax2.set_xscale('linear')
ax2.set_yscale('linear')
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True)

l_obj, = ax2.plot(x, y_obj, 'k-', alpha=0.5, lw=2.5)
l_sur, = ax2.plot(x, y_sur, 'b--', lw=2.5)
pc = ax2.fill_between(x, y_sur-y_sigma, y_sur+y_sigma, color=(0, 0, 1, 0.2))
vl2_acq = ax2.axvline(bo.x_evals[-1], ls='--', c='k', lw=2.5)
l_bo_evals, = ax2.plot(bo.x_evals, bo.y_evals, 'b.', ms=20, alpha=0.5)
l_bo_opt, = ax2.plot(bo.x_opt, bo.y_opt, 'bv', ms=20, label='BO')

l_nm_evals, = ax2.plot(nm.x_evals, nm.y_evals, 'r.', ms=20, alpha=0.5)
l_nm_opt, = ax2.plot(nm.x_opt, nm.y_opt, 'rv', ms=20, label='NM')

ax2.plot(x0, sim.to_objective_fun(x0), 'kv', ms=20, label='Initial')
ax2.legend(fontsize=20)

max_evals = 20
for num_evals in range(3, max_evals+1):

    # acq
    l_acq.set_ydata(bo.acquisition_fun(x.reshape(-1, 1)).reshape(-1))
    ax1.set_ylim(min(l_acq.get_ydata()), max(l_acq.get_ydata()))

    # bo run, sur
    bo.run(1)
    y_sur, y_sigma = bo.surrogate_fun(x.reshape(-1, 1))
    y_sur = y_sur.reshape(-1)
    l_sur.set_ydata(y_sur)
    pc.remove()
    pc = ax2.fill_between(x, y_sur - y_sigma, y_sur + y_sigma, color=(0, 0, 1, 0.2))
    l_bo_evals.set_data(bo.x_evals, bo.y_evals)
    vl1_acq.set_xdata(bo.x_evals[-1])
    vl2_acq.set_xdata(bo.x_evals[-1])
    l_bo_opt.set_data(bo.x_opt, bo.y_opt)

    # nm run
    nm.run(1)
    l_nm_evals.set_data(nm.x_evals, nm.y_evals)
    l_nm_opt.set_data(nm.x_opt, nm.y_opt)

    fig.suptitle('Evals.: {}, BO: {:.2e}, NM: {:.2e}'.format(num_evals, bo.y_opt, nm.y_opt), fontsize=20)
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()
