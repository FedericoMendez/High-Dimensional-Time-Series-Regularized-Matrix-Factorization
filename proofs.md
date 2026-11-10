# Proofs of some formulas used during the project

## Proof of the expression of the update of F[i]

Let $L(f_i)=\sum_{(i,t)\in \Omega}(Y_{it}-f_i^\top x_t)^2+\lambda_f ||F||^2$ the `F_step_function()`

We have :
$$\frac{\partial L(f_i)}{\partial F_{ij}}=-2\sum_t (Y_{it}-f_i^\top x_t)X_{jt}+2\lambda_f F_{ij}$$

Then :
$\begin{align}\frac{\partial L(f_i)}{\partial F_{ij}}=0&\Rightarrow \sum_t (Y_{it}-f_i^\top x_t)X_{jt} = \lambda_f F_{ij}\\
&\Rightarrow \sum_{t} (Y_{it}-\sum_l F_{il}X_{lt})X_{jt}=\lambda_f F_{ij}\\
&\Rightarrow -\sum_{t}F_{ij}X_{jt}^2+\sum_{t} \underbrace{(Y_{it}-\sum_{l\neq j}F_{il}X_{lt})}_{R_{ijt}}X_{jt}=\lambda_f F_{ij}\\
&\Rightarrow F_{ij}=\frac{\sum_t R_{ijt} X_{jt}}{\lambda_f+\sum_t X_{jt}^2}\end{align}$

## Update of W

The function to minimize is : $$\begin{align*}f(\bar w)&=\sum_{t=m}^T\left(x_t-\sum_{l\in \mathcal L}w_lx_{t-l}\right)^2+\frac{\lambda_W}{\lambda_X}||\bar w||^2\\&=\sum_{t=m}^T\left(x_t-\sum_{l=1}^Lw_lx_{t-l}\right)^2+\frac{\lambda_W}{\lambda_X}||\bar w||^2\end{align*}$$

Let $\tilde X = \begin{bmatrix}x_m & ... & x_{m-L}\\ \vdots & ... & \vdots \\ x_T & ... & x_{T-L}\end{bmatrix}$, we have : $\tilde X \bar w = \left(\sum_{l=1}^L w_l x_{t-l}\right)_{m\leq t\leq T}$

So, the function becomes : $$f(\bar w)=||\bar x - \tilde X\bar w||_2^2+\frac{\lambda_W}{\lambda_X}||\bar w||^2$$
Which is a Ridge Regression on $\bar w$ with $\lambda = \frac{\lambda_W}{\lambda_X}$.

$\nabla f(\bar w)=0 \Rightarrow \bar w = (\lambda I + \tilde X^\top \tilde X)^{-1}\tilde X^\top \bar x$

We will use Cholesky decomposition to solve the Ridge regression efficiently.
