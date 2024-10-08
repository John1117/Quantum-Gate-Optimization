import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def matrix_dot(matrix_i, matrix_j):
    return abs(np.trace(matrix_i.conj().T @ matrix_j))


def test_infidelity_of(final_gate, target_gate):
    infidelity = 1 - matrix_dot(final_gate, target_gate) ** 2 / (2 * matrix_dot(final_gate, final_gate))
    return max(infidelity, 1e-15)


def to_2d_arr(arr):
    arr_len = 1
    if arr.ndim == 0:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        arr_len = arr.shape[0]
    return arr, arr_len


class SinPulseQuantGateSimulator(object):

    def __init__(
            self,
            target_gate=np.array([[0, 1], [1, 0]], np.complex128),
            gate_time=1,
            z_const=0,
            z_noise_sigma=0,
            sin_freqs=np.array([1]),
            sin_amps=None,
            sin_amps_bounds=np.array([[-4, 4]], np.float64),
            rk45_rtol=1e-7,
            rk45_atol=1e-8,
            grid_sampling_sigmas=np.array([-1.2, 0, 1.2], np.float64),
            dtype=np.float64
    ):
        self.target_gate = target_gate

        self.gate_time = gate_time
        self.z_const = z_const
        self.z_noise = 0
        self.z_noise_sigma = z_noise_sigma
        self.sin_freqs = sin_freqs
        self.num_sin = sin_freqs.shape[0]
        if sin_amps is None:
            self.sin_amps = np.zeros(self.num_sin)
        else:
            self.sin_amps = sin_amps
        if sin_amps_bounds.shape[0] == 1:
            self.sin_amps_bounds = np.repeat(sin_amps_bounds, self.num_sin, axis=0)
        else:
            self.sin_amps_bounds = sin_amps_bounds

        self.rk45_rtol = rk45_rtol
        self.rk45_atol = rk45_atol

        self.grid_sampling_sigmas = grid_sampling_sigmas
        self.z_noises = np.linspace(
            start=self.grid_sampling_sigmas[0],
            stop=self.grid_sampling_sigmas[-1],
            num=self.grid_sampling_sigmas.shape[0]
        ) * self.z_noise_sigma

        self.dtype = dtype

    def dudt(self, t, u):
        u = u.reshape(2, 2)
        sigma_z = np.array([[1, 0], [0, -1]], self.dtype) + 0j
        sigma_x = np.array([[0, 1], [1, 0]], self.dtype) + 0j

        z_coef = self.z_const + self.z_noise
        x_coef = np.sum(self.sin_amps * np.sin(self.sin_freqs * np.pi * t / self.gate_time))
        derivitive = -0.5j * np.pi * (z_coef * sigma_z + x_coef * sigma_x) @ u
        return derivitive.reshape(4)

    def run(self):
        sol = solve_ivp(
            fun=self.dudt,
            t_span=(0, self.gate_time),
            y0=(np.identity(2, self.dtype) + 0j).reshape(4),
            t_eval=(self.gate_time,),
            method='RK45',
            rtol=self.rk45_rtol,
            atol=self.rk45_atol
        )
        final_gate = sol.y.reshape(2, 2)
        return final_gate

    def to_objective_fun(self, sin_amps):
        sin_amps, num_evals = to_2d_arr(sin_amps)
        num_noises = self.z_noises.shape[0]
        infidelities = np.zeros((num_evals, num_noises))
        for idx_evals in range(num_evals):
            self.sin_amps = sin_amps[idx_evals]
            for idx_noises in range(num_noises):
                self.z_noise = self.z_noises[idx_noises]
                final_gate = self.run()
                infidelities[idx_evals, idx_noises] = test_infidelity_of(final_gate, self.target_gate)
        return infidelities.mean(axis=1, keepdims=True)


if __name__ == '__main__':
    h = np.array([[1, 1], [1, -1]], np.complex128)*2**-0.5
    x = np.array([[0, 1], [1, 0]], np.complex128)
    sim = SinPulseQuantGateSimulator(
        target_gate=x,
        gate_time=1,
        z_const=0,
        sin_amps_bounds=np.array([[-4, 4]], np.float64),
        rk45_rtol=1e-7,
        rk45_atol=1e-8,
        sin_freqs=np.array([1, 3, 5]),
        grid_sampling_sigmas=np.array([0], np.float64)
    )

    r = minimize(
        fun=sim.to_objective_fun,
        x0=np.random.uniform(sim.sin_amps_bounds[:, 0], sim.sin_amps_bounds[:, 1], sim.sin_amps_bounds.shape[0]),
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15}
    )
    g = sim.run()
    print(g)
    print(r.fun)
