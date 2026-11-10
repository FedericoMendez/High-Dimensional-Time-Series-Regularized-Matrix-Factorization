import numpy as np

class TRMF_AR:
    """This class implements the TRMF-AR method from the paper "Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction"."""
    def __init__(self, Y, latent_dim, lag_set, sigma, eta, lambda_x, lambda_f, lambda_W):
        self.Y=Y # Observation

        # Parameters
        self.n, self.T = Y.shape[0], Y.shape[1]
        self.k = latent_dim
        self.L_set = lag_set; self.L = max(self.L_set); self.m = self.L+1
        self.sigma_noise = sigma
        self.lambda_x, self.lambda_f, self.lambda_W = lambda_x, lambda_f, lambda_W
        self.eta = eta

        # Variables
        self.X = self.random.random(self.k, self.T)
        self.F = self.random.random(self.n, self.k)
        self.W = self.random.random(self.k, self.L); self.W = self.W / np.sum(self.W, axis=0, keepdims=True)
        pass

    def model_W(self, prev_x):
        """Calculates x_t based on the current AR model and the previous element x_{t-1}."""
        sum=0
        for l in range(len(self.L_set)):
            sum+= self.W[:,l]@prev_x
        eps_t = np.random.multivariate_normal(np.zeros(self.k), self.sigma_noise * np.eye(self.k))
        return sum + eps_t

    # Regularization functions
    def Reg(self, M):
        return np.linalg.norm(M)**2

    def T_M(self):
        # X has shape (k, T) so we do the norm on the columns
        first_term=np.sum([np.linalg.norm(self.X[:,t]-np.sum([self.W[:,l]@self.X[:,t-l] for l in self.L_set])) for t in range(self.L+1, self.T+1)])
        return first_term + (self.eta/2)*np.linalg.norm(self.X, axis=1)**2

    def optimization_function(self):
        return np.linalg.norm(self.Y-self.F@self.X)**2 + self.lambda_x * self.Rx(self.X) + self.lambda_f * self.Reg(self.F) + self.lambda_W * self.Reg(self.W)

    # Alternative update functions
    def F_step_func(self, F):    
        return np.linalg.norm(self.Y-F@self.X)**2 + self.lambda_f * self.reg(F)

    def W_step_func(self, W):
        new_W = np.zeros_like(W)
        for r in range(W.shape[0]):
            x = self.X[r]
            x_target = x[self.m:]
            X_tilde = np.array([[x[self.m+i-1-j] for j in range(self.L)]for i in range(self.T-self.m+1)])
            
            # Solving Ridge Regression with Cholesky decomposition
            A = X_tilde.T@X_tilde + (self.lambda_W/self.lambda_X)*np.eye(self.L)
            L = np.linalg.cholesky(A)
            z= np.linalg.solve(L, X_tilde.T @ x_target)
            new_W[r]= np.linalg.solve(L.T, z)

        return new_W

    def coordinate_descent(self, F, max_steps = 200, eps=1e-4, grad_step=1e-5):
        k=0
        curr_F = np.array(F)
        denom = np.sum(self.X**2, axis=1)+self.lambda_f

        for k in range(max_steps):
            old_F = np.copy(curr_F)
            for i in range(curr_F.shape[0]):
                # Calculating the gradient w.r.t F_i, we can find its formula.
                # The proof of this final formula is in proofs.md
                for j in range(curr_F.shape[1]):
                    fx_without_j = np.dot(curr_F[i], self.X)-curr_F[i,j]*self.X[j,:]
                    R_ij = self.Y[i,:]-fx_without_j
                    curr_F[i,j]=np.dot(R_ij, self.X[j,:])/denom[j]

            # Convergence conditions    
            if np.linalg.norm(curr_F-old_F)<= eps:
                break
        return curr_F
    
    def X_step_func(self, X):
        return  

    def W_step_func(self):
        return np.linalg.norm(self.W)**2

    def optimize(self):
        # Update F - Alternative Least Square / Coordinate descent
        self.F = self.coordinate_descent(self.F)
        # Update X - GRALS based on theorem 1
        
        # Update W - Cholesky Factorization
        self.W = self.W_step_func(self.W)
        return