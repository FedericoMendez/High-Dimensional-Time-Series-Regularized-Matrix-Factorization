import numpy as np

class TRMF_AR:
    def __init__(self, Y, latent_dim, lags, sigma, lambda_x, lambda_f, lambda_W):
        self.Y=Y
        self.n, self.T = Y.shape[0], Y.shape[1]
        self.k = latent_dim
        self.L_set = lags; self.L = max(self.L_set)
        self.sigma_noise = sigma
        self.lambda_x, self.lambda_f, self.lambda_W = lambda_x, lambda_f, lambda_W
        self.F = self.random.random(self.n, self.k)
        self.W = self.random.random(self.k, self.L); self.W = self.W / np.sum(self.W, axis=0, keepdims=True)
        pass

    def model_theta(self, prev_x):
        sum=0
        for l in range(len(self.L_set)):
            sum+= self.W[:,l]@prev_x
        eps_t = np.random.multivariate_normal(np.zeros(self.k), self.sigma_noise * np.eye(self.k))
        return sum + eps_t

    def Rx(self, X):
        return 
    
    def Rf(self, F):
        return 

    def R_W(self):
        return np.linalg.norm(self.W)**2

    def optimization_function(self, X):
        return np.linalg.norm(self.Y-self.F@X)**2 + self.lambda_x * self.Rx(X) + self.lambda_f * self.Rf(self.F) + self.lambda_W * self.R_W(self.W)