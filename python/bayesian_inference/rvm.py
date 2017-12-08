import numpy as np
import scipy.optimize as opt

class RVM:

    def __init__(self, x_samples, labels):
        self.x_samples = x_samples
        self.labels = labels

        # Use all functions for basis function kernels
        self.update_x_m(self.x_samples)

        # Initialize
        sz = (len(x_samples), 1)
        self.w = np.zeros(sz)
        self.b = 0
        self.alpha = np.ones(sz)

    def basis_function(self, x, x_m, r=0.5):
        return np.exp(-np.square(np.linalg.norm(x-x_m))/r**2)

    def compute_phi(self, xs):
        [n1, d] = xs.shape
        [n2, d] = self.x_m.shape

        phi = np.ones((len(xs), len(self.x_m)))

        for i in range(n1):
            x_i = xs[i, :]
            for j in range(n2):
                x_mj = self.x_m[j, :]
                phi[i, j] = self.basis_function(x_i, x_mj)

        return phi

    def update_x_m(self, x_m):
        self.x_m = x_m
        self.phi_samples = self.compute_phi(self.x_samples)

    def predict_labels(self, samples):
        phi = self.compute_phi(samples)
        val = np.dot(phi, self.w) + self.b

        return np.squeeze(1/(1+np.exp(-val)))

    def compute_log_likelihood(self, w = None, b = None):
        if w is None:
            w = self.w

        if b is None:
            b = self.b

        y = np.dot(self.phi_samples, w) + b
        y_n = 1/(1+np.exp(-y))

        c1 = np.sum(np.multiply(self.labels, np.log(y_n)))
        c2 = np.sum(np.multiply(1 - self.labels, np.log(1-y_n)))

        if len(self.alpha)>1:
            A = np.diag(np.squeeze(self.alpha))
        else:
            A = np.squeeze(self.alpha)*np.eye(1)

        wT = w.transpose()
        c3 = -0.5*np.dot(np.dot(wT, A), w)

        return np.squeeze(c1+c2+c3)


    def update_w_mp(self):
        def cost(x):
            #return -self.compute_log_likelihood(x[:-1], x[-1])
            return -self.compute_log_likelihood(x[:-1])

        res = opt.minimize(cost, np.append(self.w, self.b))
        params_mp = np.reshape(res.x, (len(self.w)+1, 1))
        w_mp = params_mp[:-1]
        b_mp = params_mp[-1]

        print 'll new', self.compute_log_likelihood(w_mp, b_mp)
        print 'll old', self.compute_log_likelihood(self.w, self.b)

        self.w = w_mp
        self.b = b_mp

        #print self.w
        print res.message
        print 'Success', res.success

    def update_alpha(self):
        y = np.dot(self.phi_samples, self.w) + self.b
        y_n = 1/(1+np.exp(-y))

        b = np.multiply(y_n, 1 - y_n)
        B = np.diag(np.squeeze(b))

        A = np.diag(np.squeeze(self.alpha))

        p = self.phi_samples
        pT = p.transpose()
        h = -(np.dot(np.dot(pT, B), p) + A)

        cov = -np.linalg.inv(h)
        #print 'cov', cov

        n_x_m = len(self.x_m)
        gamma = np.zeros((n_x_m, 1))

        for i in range(n_x_m):
            gamma[i, 0] = 1 - self.alpha[i, 0]*cov[i, i]

        self.alpha = gamma / np.square(self.w)
        print 'gamma', gamma.shape
        print 'self w', self.w.shape
        print 'alpha', self.alpha.shape

        not_too_large = np.squeeze(self.alpha) < 1e3
        self.alpha = self.alpha[not_too_large, :]
        self.w = self.w[not_too_large, :]

        self.x_m = self.x_m[not_too_large, :]
        self.phi_samples = self.phi_samples[:, not_too_large]
        #self.update_x_m(self.x_m[not_too_large])

        print 'gamma', gamma.shape
        print 'alpha', self.alpha.shape

    def update_params(self, iters = 1):
        for i in range(iters):
            print 'Iteration %d / %d' % (i, iters)
            self.update_w_mp()
            self.update_alpha()
