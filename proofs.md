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
