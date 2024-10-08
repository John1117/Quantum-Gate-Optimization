import numpy as np
from simulator import SinPulseQuantGateSimulator


class NelderMeadOptimizer(object):

    def __init__(
            self,
            objective_fun,
            bounds,
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
            y_opt_threshold=1e-15
    ):
        self.objective_fun = objective_fun
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.max_fun_evals = max_fun_evals
        self.coef_reflect = coef_reflect
        self.coef_expand = coef_expand
        self.coef_contract = coef_contract
        self.coef_shrink = coef_shrink
        self.x_tol = x_tol
        self.y_tol = y_tol
        self.adapt = adapt
        self.y_opt_threshold = y_opt_threshold

        # Initial evaluation points
        if x0 is None:
            self.x_evals = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
            self.y_evals = self.objective_fun(self.x_evals)
        else:
            self.x_evals = x0
            if y0 is None:
                self.y_evals = self.objective_fun(self.x_evals)
            else:
                self.y_evals = y0

        self.x_sim, self.y_sim = self.order(self.x_evals, self.y_evals)
        self.x_o = None
        self.x_r = None
        self.y_r = None
        self.num_fun_evals = self.x_evals.shape[0]

        self.x_opt = self.x_sim[0].squeeze()
        self.y_opt = self.y_sim[0].squeeze()
        self.status = 'Reflection'
        self.idx_shrink = self.dim

    def run(self, num_runs=1):
        idx_runs = 0
        while idx_runs < num_runs:

            if self.num_fun_evals >= self.max_fun_evals:
                self.status = 'Termination'
                break

            if self.y_opt <= self.y_opt_threshold:
                self.status = 'Termination'
                break

            if self.check_convergence():
                self.status = 'Termination'
                break

            if self.status == 'Reflection':
                self.x_o = self.get_centroid()
                self.x_r = self.clip_to_bounds(self.reflect())
                self.y_r = self.objective_fun(self.x_r)
                self.stack_evals(self.x_r, self.y_r)
                if self.num_fun_evals >= self.max_fun_evals:
                    break
                idx_runs += 1

                if self.y_r <= self.y_sim[0]:
                    self.status = 'Expansion'
                elif self.y_r < self.y_sim[-2]:
                    self.replace_worst(self.x_r, self.y_r)
                    self.status = 'Order'
                else:
                    self.status = 'Contraction'

            elif self.status == 'Expansion':
                x_e = self.clip_to_bounds(self.expand(self.x_r))
                y_e = self.objective_fun(x_e)
                self.stack_evals(x_e, y_e)
                if self.num_fun_evals >= self.max_fun_evals:
                    break
                idx_runs += 1

                if y_e < self.y_r:
                    self.replace_worst(x_e, y_e)
                else:
                    self.replace_worst(self.x_r, self.y_r)
                self.status = 'Order'

            elif self.status == 'Contraction':
                if self.y_r < self.y_sim[-1]:
                    x_c = self.clip_to_bounds(self.contract(self.x_r))
                    y_c = self.objective_fun(x_c)
                    self.stack_evals(x_c, y_c)
                    if self.num_fun_evals >= self.max_fun_evals:
                        break
                    idx_runs += 1

                    if y_c <= self.y_r:
                        self.replace_worst(x_c, y_c)
                        self.status = 'Order'
                    else:
                        self.status = 'Shrink'
                else:
                    x_c = self.clip_to_bounds(self.contract(self.x_sim[-1]))
                    y_c = self.objective_fun(x_c)
                    self.stack_evals(x_c, y_c)
                    if self.num_fun_evals >= self.max_fun_evals:
                        break
                    idx_runs += 1

                    if y_c < self.y_sim[-1]:
                        self.replace_worst(x_c, y_c)
                        self.status = 'Order'
                    else:
                        self.status = 'Shrink'
            
            elif self.status == 'Shrink':
                if self.idx_shrink == 0:
                    self.idx_shrink = self.dim
                    self.status = 'Order'
                else:
                    self.x_sim[self.idx_shrink] = self.clip_to_bounds(self.shrink(self.idx_shrink))
                    self.y_sim[self.idx_shrink] = self.objective_fun(self.x_sim[self.idx_shrink])
                    self.stack_evals(self.x_sim[self.idx_shrink], self.y_sim[self.idx_shrink])
                    if self.num_fun_evals >= self.max_fun_evals:
                        break
                    self.idx_shrink -= 1
                    idx_runs += 1

            elif self.status == 'Order':
                self.x_sim, self.y_sim = self.order(self.x_sim, self.y_sim)
                self.status = 'Reflection'

            self.x_opt = self.x_sim[0].squeeze()
            self.y_opt = self.y_sim[0].squeeze()
        return self.x_opt, self.y_opt
                
    def order(self, x, y):
        idx = np.argsort(y, axis=0)[:self.dim + 1, 0]
        return x[idx], y[idx]

    def clip_to_bounds(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def stack_evals(self, x, y):
        self.x_evals = np.vstack((self.x_evals, x))
        self.y_evals = np.vstack((self.y_evals, y))
        self.num_fun_evals += 1

    def check_convergence(self):
        return np.max(np.abs(self.x_sim[1:] - self.x_sim[0])) <= self.x_tol and \
               np.max(np.abs(self.y_sim[1:] - self.y_sim[0])) <= self.y_tol

    def replace_worst(self, x, y):
        self.x_sim[-1] = x
        self.y_sim[-1] = y

    def get_centroid(self):
        return np.mean(self.x_sim[:-1], axis=0)

    def reflect(self):
        return self.x_o + self.coef_reflect * (self.x_o - self.x_sim[-1])

    def expand(self, x_r):
        return self.x_o + self.coef_expand * (x_r - self.x_o)

    def contract(self, x):
        return self.x_o + self.coef_contract * (x - self.x_o)

    def shrink(self, idx_simplex):
        return self.x_sim[0] + self.coef_shrink * (self.x_sim[idx_simplex] - self.x_sim[0])


if __name__ == '__main__':
    target_x_gate = np.array([[0, 1], [1, 0]], np.complex128)
    ideal_x_gate_simulator = SinPulseQuantGateSimulator(
        target_gate=target_x_gate,
        gate_time=1,
        z_const=0,
        z_noise_sigma=0,
        sin_freqs=np.array([1]),
        sin_amps=None,
        sin_amps_bounds=np.array([[-4, 4]], np.float64),
        rk45_rtol=1e-7,
        rk45_atol=1e-8,
        grid_sampling_sigmas=np.array([0], np.float64),
        dtype=np.float64
    )

    nmo = NelderMeadOptimizer(
        objective_fun=ideal_x_gate_simulator.to_objective_fun,
        bounds=ideal_x_gate_simulator.sin_amps_bounds,
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
        y_opt_threshold=1e-15
    )

    e = nmo.num_fun_evals
    s = nmo.status
    print('e', e)
    print('s', s, '\n')
    for i in range(50):
        x, y = nmo.run()
        e = nmo.num_fun_evals
        s = nmo.status
        print('e', e)
        print('x', x)
        print('y', y)
        print('s', s, '\n')

