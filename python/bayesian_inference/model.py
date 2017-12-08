import numpy as np
import scipy.optimize as opt

class Model:

    def __init__(self, x_samples, y_samples, x_m):
        self.x_samples = x_samples
        self.y_samples = y_samples
        self.x_m = x_m

        self.phi_samples = self.compute_phi(x_samples)

    def update_x_m(self, x_m):
        self.x_m = x_m
        self.phi_samples = self.compute_phi(self.x_samples)

    def basis_function(self, x, x_m = 0.0, r = 1.0):
        return np.exp(-np.square(x - x_m) / r**2)

    def compute_phi(self, xs):
        phi = np.zeros((len(xs), len(self.x_m)))
        for i in range(len(xs)):
            x_i = xs[i]
            for j in range(len(self.x_m)):
                x_mj = self.x_m[j]
                phi[i, j] = self.basis_function(x_i, x_m = x_mj)

        return phi

    def compute_least_squares_fit(self, l2_reg = 0):
        pTp = np.dot(self.phi_samples.transpose(), self.phi_samples)
        A = pTp + l2_reg * np.eye(len(self.x_samples))
        A_inv = np.linalg.inv(A)
        w_ls = np.dot(np.dot(A_inv, self.phi_samples.transpose()), self.y_samples)

        self.w = w_ls

    def compute_posterior(self, stddev = 0.0, alpha = 0, sparse=False):
        if sparse:
            n = len(self.x_m)
            pTp = np.dot(self.phi_samples.transpose(), self.phi_samples)

            A = np.diag(alpha)

            self.sigma = np.linalg.inv(stddev**(-2)*pTp + A)
            self.mu = stddev**-2 * np.dot(np.dot(self.sigma, self.phi_samples.transpose()), self.y_samples)
            return
        else:
            pTp = np.dot(self.phi_samples.transpose(), self.phi_samples)
            A = pTp + stddev**2 * alpha * np.eye(pTp.shape[0])
            A_inv = np.linalg.inv(A)

            self.mu = np.dot(np.dot(A_inv, self.phi_samples.transpose()), self.y_samples)
            self.sigma = stddev**2 * A_inv

    def eval_log_likelihood(self, stddev = 0.0, alpha = 1.0, sparse=False):
        if sparse:
            n = len(self.x_samples)

            p = self.phi_samples
            pT = self.phi_samples.transpose()

            A = np.diag(alpha)
            A_inv = np.linalg.inv(A)

            sigma = (stddev**2)*np.eye(n) + np.dot(np.dot(p, A_inv), pT)
            sigma_inv = np.linalg.inv(A)
            sign, log_det = np.linalg.slogdet(sigma)

            s = -n/2.0 * np.log(2*np.pi) - 1.0/2.0 * log_det

            yTAinvy = np.dot(np.dot(self.y_samples.transpose(), sigma_inv), self.y_samples)
            x = -1.0/2.0 * yTAinvy

            return s + x
        else:
            n = len(self.x_samples)
            ppT = np.dot(self.phi_samples, self.phi_samples.transpose())
            A = (stddev**2)*np.eye(n) + 1.0/alpha * ppT
            A_inv = np.linalg.inv(A)

            sign, log_det = np.linalg.slogdet(A)

            s = -n/2.0 * np.log(2*np.pi) - 1.0/2.0 * log_det

            yTAinvy = np.dot(np.dot(self.y_samples.transpose(), A_inv), self.y_samples)
            x = -1.0/2.0 * yTAinvy

            return s + x

    def get_params_mp(self, stddev0=1, alpha0=1, sparse=False, iterations=10):
        if sparse:
            n = len(self.x_samples)
            n_xm = len(self.x_m)

            var = stddev0 ** 2

            alpha = alpha0
            alpha_new = np.zeros((n_xm))

            for it in range(iterations):
                stddev = (var)**(0.5)

                self.compute_posterior(stddev, alpha, sparse=True)

                gamma = np.zeros((n_xm))

                for i in range(n_xm):
                    gamma[i] = 1 - alpha[i]*self.sigma[i, i]

                for i in range(n_xm):
                    if np.abs(self.mu[i]) < 1e-5:
                        alpha_new[i] = 1e10
                    else:
                        alpha_new[i] = gamma[i] / (self.mu[i]**2)

                diff = self.y_samples - np.dot(self.phi_samples, self.mu)
                var_new = np.linalg.norm(diff)**2 / (n - np.sum(gamma))

                alpha = alpha_new
                var = var_new

                #print 'Gamma', gamma
                #print 'Alpha', alpha
                #print 'Stddev', var**0.5

            return var**(0.5), alpha

        else:
            # Define cost function
            def cost(x):
                return -self.eval_log_likelihood(stddev = x[0], alpha = x[1])
            res = opt.minimize(cost, [stddev0, alpha0])

            return res.x

    def get_normalized_error(self, x_val, y_val):
        phi = self.compute_phi(x_val)

        y_fit = np.dot(phi, self.w)
        err = np.sum(np.square(y_fit - y_val)) / len(y_val)

        return err

    def get_map_fit(self, x_eval):
        phi = self.compute_phi(x_eval)
        return np.dot(phi, self.w)

    def sample_fit(self, x_eval, num=1):
        phi = self.compute_phi(x_eval)

        # Sample a model fit
        w = np.random.multivariate_normal(self.mu, self.sigma, size=num)
        y_evals = np.dot(phi, w.transpose())

        return y_evals

    def get_bounds(self, x_eval, stddev = 0.0, n = 1):
        phi = self.compute_phi(x_eval)
        f = phi.transpose()

        mu = np.dot(phi, self.mu)

        fTSf = np.dot(np.dot(f.transpose(), self.sigma), f)
        cov = stddev**2 + fTSf

        std = np.sqrt(np.diag(cov))

        y_lower = mu - std*n
        y_upper = mu + std*n

        return y_lower, y_upper

    def get_fit_mean(self, x_eval):
        phi = self.compute_phi(x_eval)
        y_eval = np.dot(phi, self.mu)
        return y_eval
