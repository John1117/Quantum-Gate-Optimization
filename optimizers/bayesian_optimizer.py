import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern, ExpSineSquared
from warnings import catch_warnings, simplefilter
from simulator import SinPulseQuantGateSimulator


def to_2d_arr(arr):
    arr_len = 1
    if arr.ndim == 0:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        arr_len = arr.shape[0]
    return arr, arr_len


class BayesianOptimizer(object):

    def __init__(
            self,
            objective_fun,
            bounds,
            max_fun_evals=np.inf,
            num_initial_points=1,
            x0=None,
            y0=None,
            const_value=1.0,
            const_value_bounds=(1e-5, 1e5),
            matern_len=1.0,
            matern_len_bounds=(1e-5, 1e5),
            matern_nu=1.5,
            sin_len=1.0,
            sin_len_bounds=(1e-5, 1e5),
            sin_period=1.0,
            sin_period_bounds=(1e-5, 1e5),
            gpr_optimizer='fmin_l_bfgs_b',
            gpr_num_trials=0,
            acq_fun='EI',  # LCB/EI/PI
            acq_optimizer='L-BFGS-B',  # sampling/L-BFGS-B
            acq_optimizer_options=None,
            acq_num_trials=100,
            random_state=np.random.randint(1000),
            acq_kappa=1.96,
            acq_xi=0.01,
            noise=1e-10,
            y_opt_threshold=1e-15
    ):
        self.objective_fun = objective_fun
        self.bounds = bounds
        self.max_fun_evals = max_fun_evals
        self.acq_optimizer = acq_optimizer  # sampling/L-BFGS-B
        self.acq_optimizer_options = acq_optimizer_options
        self.acq_num_trials = acq_num_trials
        self.random_state = random_state
        self.acq_kappa = acq_kappa
        self.acq_xi = acq_xi
        self.noise = noise
        self.y_opt_threshold = y_opt_threshold

        # Initial evaluation points
        num_initial_points = 1 if num_initial_points < 1 else num_initial_points
        if x0 is None:
            self.x_evals = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_points, bounds.shape[0]))
            self.y_evals = self.objective_fun(self.x_evals)
        else:
            self.x_evals = x0
            if y0 is None:
                self.y_evals = self.objective_fun(self.x_evals)
            else:
                self.y_evals = y0

        # Surrogate function initialization
        const = ConstantKernel(const_value, const_value_bounds)
        matern = Matern(matern_len, matern_len_bounds, matern_nu)
        sin = ExpSineSquared(sin_len, sin_period, sin_len_bounds, sin_period_bounds)
        kernel = const
        if matern_len is not None:
            kernel *= matern
        if sin_len is not None:
            kernel *= sin
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise,
            optimizer=gpr_optimizer,
            n_restarts_optimizer=gpr_num_trials-1,
            random_state=random_state
        )

        self.num_fun_evals = self.x_evals.shape[0]
        idx_opt = np.argmin(self.y_evals, axis=0)
        self.x_opt = self.x_evals[idx_opt].squeeze()
        self.y_opt = self.y_evals[idx_opt].squeeze()
        self.fit_surrogate_fun_to(self.x_evals, self.y_evals)

        # Acquisition function initialization
        if acq_fun == 'EI':
            self.acquisition_fun = self.expected_improvement
        elif acq_fun == 'PI':
            self.acquisition_fun = self.probability_of_improvement
        elif acq_fun == 'LCB':
            self.acquisition_fun = self.lower_confidence_bound

    def run(self, num_runs=1):
        idx_runs = 0
        while idx_runs < num_runs:

            if self.num_fun_evals >= self.max_fun_evals:
                break

            if self.y_opt <= self.y_opt_threshold:
                break

            x_acq = self.acquire_x()
            y_acq = self.objective_fun(x_acq)

            self.x_evals = np.vstack((self.x_evals, x_acq))
            self.y_evals = np.vstack((self.y_evals, y_acq))
            self.fit_surrogate_fun_to(self.x_evals, self.y_evals)

            idx_runs += 1
            self.num_fun_evals += 1
            if y_acq < self.y_opt:
                self.x_opt = x_acq.squeeze()
                self.y_opt = y_acq.squeeze()

        return self.x_opt, self.y_opt

    def surrogate_fun(self, x):
        x, _ = to_2d_arr(x)
        with catch_warnings():
            simplefilter('always')
            mu, sigma = self.gpr.predict(x, return_std=True)
        return mu, sigma

    def fit_surrogate_fun_to(self, x, y):
        x, _ = to_2d_arr(x)
        y, _ = to_2d_arr(y)
        with catch_warnings():
            simplefilter('always')
            self.gpr.fit(x, y)

    def acquire_x(self):
        x_acq = None
        val_acq = -np.inf
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.acq_num_trials, self.bounds.shape[0])):
            if self.acq_optimizer == 'sampling':
                x = x0
                val = self.acquisition_fun(x0)
            else:
                res = minimize(
                    fun=lambda t: -self.acquisition_fun(t).squeeze(),
                    x0=x0,
                    bounds=self.bounds,
                    method=self.acq_optimizer,
                    options=self.acq_optimizer_options
                )
                x = res.x
                val = -res.fun
            if val > val_acq:
                x_acq = x
                val_acq = val
        return x_acq

    def expected_improvement(self, x):
        x, _ = to_2d_arr(x)
        mu, sigma = self.surrogate_fun(x)
        sigma = sigma.reshape(-1, 1)

        imp = self.y_opt - mu - self.acq_xi
        with np.errstate(divide='ignore'):
            z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma <= 0.0] = 0.0
        return ei

    def probability_of_improvement(self, x):
        x, _ = to_2d_arr(x)
        mu, sigma = self.surrogate_fun(x)
        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide='ignore'):
            z = (self.y_opt - mu - self.acq_xi) / sigma
        pi = norm.cdf(z)
        pi[sigma <= 0.0] = 0.0
        return pi

    def lower_confidence_bound(self, x):
        x, _ = to_2d_arr(x)
        mu, sigma = self.surrogate_fun(x)
        sigma = sigma.reshape(-1, 1)

        lcb = self.acq_kappa * sigma - mu
        return lcb


if __name__ == '__main__':
    target_x_gate = np.array([[0, 1], [1, 0]], np.complex128)
    ideal_x_gate_simulator = SinPulseQuantGateSimulator(
        target_gate=target_x_gate,
        gate_time=1,
        z_const=0,
        z_noise_sigma=0,
        sin_freqs=np.array([1, 3]),
        sin_amps=None,
        sin_amps_bounds=np.array([[-4, 4]], np.float64),
        rk45_rtol=1e-7,
        rk45_atol=1e-8,
        grid_sampling_sigmas=np.array([0], np.float64),
        dtype=np.float64
    )

    bo = BayesianOptimizer(
        objective_fun=ideal_x_gate_simulator.to_objective_fun,
        bounds=ideal_x_gate_simulator.sin_amps_bounds,
        max_fun_evals=np.inf,
        num_initial_points=1,
        x0=None,
        y0=None,
        const_value=1,
        const_value_bounds=(1e-5, 1e5),
        matern_length_scale=1,
        matern_length_scale_bounds=(1e-5, 1e5),
        matern_nu=1.5,
        gpr_optimizer='fmin_l_bfgs_b',
        gpr_num_trials=3,
        acq_fun='EI',  # LCB/EI/PI
        acq_optimizer='Nelder-Mead',  # sampling/L-BFGS-B
        acq_num_trials=50,
        random_state=np.random.randint(1000),
        acq_kappa=1.96,
        acq_xi=0.01,
        noise=1e-10,
    )

    e = bo.num_fun_evals
    print('e', e, '\n')
    for i in range(10):
        x, y = bo.run()
        e = bo.num_fun_evals
        print('e', e)
        print('x', bo.x_evals)
        print('y', bo.y_evals, '\n')
