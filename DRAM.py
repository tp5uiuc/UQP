from dataclasses import dataclass
import numpy as np
from scipy.stats import multivariate_normal as mvn_dist
from tqdm.notebook import trange

@dataclass
class DRAMParameters:
    M: int
    k_0: int
    # burn_in_pc: int
    # sigma_squared: float
    n_s: float = 0.1


class SimulationResults:
    def __init__(self):
        self.results = {}
    
class DRAM:
    def __init__(self, params: DRAMParameters):
        self.params = params
        self.simulation_results = SimulationResults()
        # self.sample_history = list()
        self.first_stage_acceptance = 0
        self.second_stage_acceptance = 0

    def __call__(
        self, residual_gen, initial_mean, initial_variance, initial_covariance
    ):
        rng = np.random.default_rng()

        q_prev = initial_mean.copy()
        cov_prev = initial_covariance.copy()
        variance_prev = initial_variance.copy()
        sigma_squared = initial_variance.copy()

        R = np.linalg.cholesky(cov_prev)
        np.testing.assert_allclose(cov_prev, R @ R.T)

        n = residual_gen(q_prev).size
        p = q_prev.size

        residual_ss_gen = lambda x: np.linalg.norm(residual_gen(x), 2)
        ss_q_prev = residual_ss_gen(q_prev)

        zero_p = np.zeros((p,))
        I_p = np.eye(p)
        s_p = 2.38 / (p ** 2)
        gamma_2 = 1.0 / 5.0

        # prepare for adaptive metropolis
        # k_sample_history = []
        # k_sample_history.append(q_prev)
        
        sample_history = list()
        s2_history = list()
        sample_history.append(q_prev)
        s2_history.append(variance_prev)
        
        # self.results['chain'] = []
        # self.results['chain'].append(q_prev)

        for curr in trange(1, self.params.M + 1):
            # (a)
            z_curr = rng.multivariate_normal(zero_p, I_p)

            # (b)
            q_star = q_prev + R @ z_curr

            # (c)
            uniform_sample = rng.random()

            # step (d)
            ss_q_star = residual_ss_gen(q_star)

            # step (e)
            threshold = np.exp(-(ss_q_star - ss_q_prev) / 2.0 / variance_prev)
            alpha_qstar_upon_q = np.minimum(1.0, threshold)

            # step(f)
            if np.less(uniform_sample, alpha_qstar_upon_q):
                q_prev = q_star.copy()
                ss_q_prev = ss_q_star.copy()
                self.first_stage_acceptance += 1
            else:
                # Algorithm 8.10
                """
                No delayed rejection
                """
                

                """
                Delayed rejection
                """
                z_delayed = rng.multivariate_normal(zero_p, I_p)
                q_star2 = q_prev + gamma_2 * R @ z_curr
                uniform_sample_delayed = rng.random()
                ss_q_star2 = residual_ss_gen(q_star2)

                qstar_upon_qstar2_threshold = np.exp(
                    -(ss_q_star - ss_q_star2) / 2.0 / variance_prev
                )
                alpha_qstar_upon_qstar2 = np.minimum(1.0, qstar_upon_qstar2_threshold)

                first_stage_contribution = (1.0 - alpha_qstar_upon_qstar2) / (
                    1.0 - alpha_qstar_upon_q
                )

                qstar2_threshold = np.exp(
                    -(ss_q_star2 - ss_q_prev) / 2.0 / variance_prev
                )
                # J(q_star | q_star2) / J(q_star | q_prev)
                pdf_ratio = mvn_dist.pdf(
                    q_star, mean=q_star2, cov=cov_prev
                ) / mvn_dist.pdf(q_star, mean=q_prev, cov=cov_prev)
                alpha_qstar2_upon_q_q_star = np.minimum(
                    1.0, qstar2_threshold * pdf_ratio * first_stage_contribution
                )

                if np.less(uniform_sample_delayed, alpha_qstar2_upon_q_q_star):
                    q_prev = q_star2.copy()
                    ss_q_prev = ss_q_star2.copy()
                    self.second_stage_acceptance += 1
                else:
                    # do nothing, similar to standard
                    pass

            # record history for adaptation
            sample_history.append(q_prev)

            # (g)
            shape_parameter = 0.5 * (self.params.n_s + n)
            # ss_q_prev is updated already
            scale_parameter = 1.0 / (
                0.5 * (self.params.n_s * sigma_squared + ss_q_prev)
            )
            g = rng.gamma(shape_parameter, scale_parameter)
            variance_prev = 1.0 / g
            s2_history.append(variance_prev)
            
            # (h)
            if not (curr % self.params.k_0):
                # adaptive metropolis
                k_sample_history_arr = np.array(sample_history)
                k_samples = k_sample_history_arr.shape[0]
                # print(curr, k_samples, self.params.k_0)
                # assert k_samples == self.params.k_0
                mean_k_samples = np.mean(k_sample_history_arr, axis=0)
                # sp cov(q0, q1, ... qk) # (k + 1 entities)
                cov_k_samples = (
                    1.0
                    / (k_samples - 1)
                    * (
                        k_sample_history_arr.T @ k_sample_history_arr
                        - (k_samples * np.outer(mean_k_samples, mean_k_samples))
                    )
                )
                cov_prev = s_p * cov_k_samples
                # q_prev here is current after update
            else:
                # standard metropolis, set k0 to be very very high
                # cov_prev = cov_prev
                pass

            # (i)
            R = np.linalg.cholesky(cov_prev)
            
        self.simulation_results.results['chain'] = np.array(sample_history)
        self.simulation_results.results['s2chain'] = np.array(s2_history)
