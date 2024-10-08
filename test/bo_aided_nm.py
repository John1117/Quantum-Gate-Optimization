import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from simulator import SinPulseQuantGateSimulator
from optimizers.bayesian_optimizer import BayesianOptimizer
from optimizers.nelder_mead_optimizer import NelderMeadOptimizer
matplotlib.use('TkAgg')

y_opt_threshold = 1e-15
x_gate = np.array([[0, 1], [1, 0]], np.complex128)
h_gate = np.array([[1, 1], [1, -1]], np.complex128)*2**-0.5
sim = SinPulseQuantGateSimulator(
    target_gate=h_gate,
    gate_time=2**-0.5*1,
    z_const=1,
    z_noise_sigma=0,
    sin_freqs=np.arange(1, 60, 2, dtype=np.float64),
    sin_amps=None,
    sin_amps_bounds=np.array([[-4, 4]], np.float64),
    rk45_rtol=1e-7,
    rk45_atol=1e-8,
    grid_sampling_sigmas=np.array([0], np.float64),
    dtype=np.float64
)

max_evals = 1000
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
nm.run(max_evals-sim.num_sin-1)

bo = BayesianOptimizer(
    objective_fun=sim.to_objective_fun,
    bounds=sim.sin_amps_bounds,
    max_fun_evals=np.inf,
    num_initial_points=1,
    x0=nm.x_evals,
    y0=nm.y_evals,
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

max_iters = 3
for idx_iters in range(1, max_iters+1):

    rdnm = NelderMeadOptimizer(
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

    bo.run(sim.num_sin+1)
    bonm = NelderMeadOptimizer(
        objective_fun=sim.to_objective_fun,
        bounds=sim.sin_amps_bounds,
        x0=bo.x_evals[-sim.num_sin-1:],
        y0=bo.y_evals[-sim.num_sin-1:],
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

    evals_lst = []
    y_rdnm_lst = []
    y_bonm_lst = []

    plt.ion()
    fig = plt.figure(num='bo_aided_nm', figsize=(15, 8), tight_layout=True)
    fig.suptitle('Iters.={}'.format(idx_iters), fontsize=20)
    ax = fig.add_subplot()
    for num_evals in range(sim.num_sin+2, max_evals+1):
        evals_lst.append(num_evals)
        y_rdnm_lst.append(rdnm.run()[1])
        y_bonm_lst.append(bonm.run()[1])

        ax.clear()
        ax.set_xlabel('Num. of Evaluations', fontsize=20)
        ax.set_ylabel('Avg. Infidelity', fontsize=20)
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True)

        ax.plot(evals_lst, y_rdnm_lst, 'r-', label='RandomInitNM', lw=5)
        ax.plot(evals_lst, y_bonm_lst, 'b-', label='BOInitNM', lw=5)
        ax.legend(loc=1, fontsize=20)

        # fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    print('Iters: ', idx_iters)
    print('Opt of RandomInitNM: ', rdnm.y_opt)
    print('Opt of BOInitNM: ', bonm.y_opt, '\n')
    if idx_iters < max_iters:
        plt.close()
plt.show()
