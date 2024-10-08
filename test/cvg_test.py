import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from simulator import SinPulseQuantGateSimulator
from optimizers.bayesian_optimizer import BayesianOptimizer
from optimizers.nelder_mead_optimizer import NelderMeadOptimizer


y_opt_threshold = 1e-15
x_gate = np.array([[0, 1], [1, 0]], np.complex128)
h_gate = np.array([[1, 1], [1, -1]], np.complex128)*2**-0.5
sim = SinPulseQuantGateSimulator(
    target_gate=h_gate,
    gate_time=2**-0.5*20,
    z_const=1,
    z_noise_sigma=0,
    sin_freqs=np.arange(1, 200, 2, dtype=np.float64),
    sin_amps=None,
    sin_amps_bounds=np.array([[-4, 4]], np.float64),
    rk45_rtol=1e-7,
    rk45_atol=1e-8,
    grid_sampling_sigmas=np.array([0], np.float64),
    dtype=np.float64
)

dim = sim.num_sin

bo = BayesianOptimizer(
    objective_fun=sim.to_objective_fun,
    bounds=sim.sin_amps_bounds,
    max_fun_evals=np.inf,
    num_initial_points=1,
    x0=None,
    y0=None,
    const_value=1.0,
    const_value_bounds=(1e-5, 1e5),
    matern_len=1.0,
    matern_len_bounds=(1e-5, 1e5),
    matern_nu=1.5,
    sin_len=None,
    sin_len_bounds=(1e-5, 1e5),
    sin_period=1.0,
    sin_period_bounds=(1e-5, 1e5),
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

num_nms = 5
nm_lst = []
for _ in range(num_nms):
    nm = NelderMeadOptimizer(
            objective_fun=sim.to_objective_fun,
            bounds=sim.sin_amps_bounds,
            x0=None,
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
    nm_lst.append(nm)

# fig
matplotlib.use('TkAgg')
plt.ion()

# fig name
fig_name = ''
if sim.z_const == 0:
    fig_name += 'x'
elif sim.z_const == 1:
    fig_name += 'h'

if sim.z_noise_sigma == 0:
    fig_name += '_ideal'
else:
    fig_name += '_noisy'

fig_name += '_{}p'.format(sim.num_sin)
fig_name += ''

# fig title
fig_title = ''
fig_title += '{}-Sin'.format(sim.num_sin)
if sim.z_noise_sigma == 0:
    fig_title += ' Ideal'
else:
    fig_title += ' Noisy'

if sim.z_const == 0:
    fig_title += ' X'
elif sim.z_const == 1:
    fig_title += ' H'
fig_title += ''

fig = plt.figure(num=fig_name, figsize=(15, 8), tight_layout=True)
fig.suptitle(fig_title, fontsize=20)
#fig.show()
ax = fig.add_subplot()

evals_lst = []
y_bo_lst = []
y_nm_lst = []

max_num_evals = int(1e5)
for num_evals in range(dim + 2, max_num_evals + 1):
    evals_lst.append(num_evals)
    y_bo_lst.append(bo.run()[-1])
    y_nm_lst.append([nm.run()[1] for nm in nm_lst])
    mu_nm = np.exp(np.mean(np.log(y_nm_lst), axis=1))

    ax.clear()
    ax.set_xlabel('Num. of Evaluations', fontsize=20)
    ax.set_ylabel('Avg. Infidelity', fontsize=20)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)

    ax.plot(evals_lst, y_bo_lst, 'b-', label='BO', lw=5)
    ax.plot(evals_lst, y_nm_lst, 'r-', alpha=0.2)
    ax.plot(evals_lst, mu_nm, 'r-', label='NM', lw=5)
    ax.legend(loc=1, fontsize=20)

    #fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()
