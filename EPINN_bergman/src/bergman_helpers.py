import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import t
from numba import njit

softplus = nn.Softplus()
ssigmoid = nn.Sigmoid()
mse_loss = nn.MSELoss()


    
class eFCN(nn.Module):
      """
      Multilayer perceptron model, with outputs corresponding to targets of Deep Evidential Regression

      Args: (int)
          N_INPUT : number of independent variables in the ODE
          N_HIDDEN : number of neurons per hidden layer
          N_LAYERS : number of hidden layers

      Returns:
          tuple (torch.Tensor)
          alpha, beta, nu, gamma : the glucose-related outputs, with gamma being the mean and the
                                   rest related to the uncertainty
                           x_var : glucose-uptake rate variable in Bergman's model; as it is 
                                   indirectly observed with no training data, we don't attempt to 
                                   learn its uncertainty to avoid overfitting
      
      """
      def __init__(self, N_INPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.output_alpha = nn.Linear(N_HIDDEN, 1)
        self.output_beta = nn.Linear(N_HIDDEN, 1)
        self.output_nu = nn.Linear(N_HIDDEN, 1)
        self.output_gamma = nn.Linear(N_HIDDEN, 1)
        self.output_X = nn.Linear(N_HIDDEN, 1)

      def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)

        alpha = 1.1 + softplus(self.output_alpha(x))
        beta = softplus(self.output_beta(x))
        nu = 0.5 + softplus(self.output_nu(x))
        gamma = softplus(self.output_gamma(x))
        x_var = ssigmoid(self.output_X(x))

        return alpha, beta, nu, gamma, x_var



def CI_calculation(interval_CI, alpha, beta, nu, gamma):
    """
    Function that computes the confidence interval for the model's output

    Args: 
        interval_CI (float): confidence interval, e.g. 0.95 for 95% interval
        alpha, beta, nu, gamma (NumPy array): evidential model's outputs with 'gamma' being mean

    Returns: 
        lower and upper bounds of the confidence interval
    """
    from scipy.stats import t
    scale_sq = beta*(1+nu)/(alpha*nu)
    scale = np.sqrt(scale_sq)

    I_delta = []

    for i in range(len(alpha)):
      a,b= t.interval(interval_CI, df=2*alpha[i], loc=0, scale=scale[i])
      I_delta.append(a)

    return np.array(I_delta)



def bergman_rhs(t, y, p, I_of_t):
    """
    Returns the differentials of G and X -- RHS of the Bergman equations.
    """
    G, X = y
    p1, p2, p3, Gb, Ib = p
    It = float(I_of_t(t))
    dG = -p1 * (G - Gb) - X * G
    dX = -p2 * X + p3 * (It - Ib)
    return [dG, dX]



def evidential_data_loss(u_obs, gamma, nu, alpha, beta ):
    """
    Loss function of Evidential Deep Learning (EDL)

    Args:
        u_obs (torch.Tensor): observed glucose
        alpha, beta, nu, gamma (torch.Tensor): evidential model's outputs with 'gamma' being mean

    Returns:
        value of EDL loss -- this is the main loss term governing glucose prediction with uncertainty estimates.
    """
    twoBlambda = 2*(beta)*(1+nu)

    nll = 0.5*torch.log(torch.pi/(nu))  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(nu*(u_obs-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return torch.mean(nll)


@njit
def bergman_likelihood_array(V, W, U, grad_G, G_pred, grad_X, X_pred, I_ext, Gb):
    """
    Computes the PDE residual for each point of the parameter domain. 
    The Bergman's model involves a pair of coupled ODEs involving variables G and X, 
    and we take the PDE residual to be the sum of both. 

    Args: (NumPy arrays) 
        V,W,U : the domains for p_1, p_2, p_3
        grad_G : time derivative of glucose
        grad_X : time derivative of X
        G_pred : model-predicted glucose
        X_pred : model-predicted X
        Gb : basal glucose level
        I_ext : insulin 

    Returns: 
        the PDE residual evaluated at each time of glucose measurement for all values of the parameter domain
    
    """
    I, J, K, M = len(V), len(W), len(U), len(grad_G)
    Z = np.empty((I, J, K, M))
    for i in range(I):
        for j in range(J):
          for k in range(K):
            p1 = V[i]
            p2 = W[j]
            p3 = U[k]
            resi_G = (grad_G + p1*(G_pred - Gb) + X_pred*G_pred )**2
            resi_X = (grad_X + p2*X_pred - p3*(I_ext))**2
            Z[i,j,k,:]= resi_G + resi_X
    return Z

   

def uni_gaussian(x, mu, std):
    """
    Univariate Gaussian function

    Args: (NumPy arrays)
        x: the Gaussian variable
        mu: mean of Gaussian
        std: standard deviation of Gaussian
    
    Returns:
           the univariate Gaussian function
    """
    return (1/np.sqrt(2*3.14*std**2))*np.exp(-1*(1/(2*std**2))*(x - mu)**2)



def kl_divergence_3D(variance_array, resi_sum_array,
                   p1_prior, p2_prior, p3_prior):
    """
    Computes the Kullback-Leibler divergence between the likelihood function and 
    and the product of the prior functions for each unknown parameter p1, p2, p3.
    
    Args: (NumPy arrays)
        variance_array: variance of the PDE likelihood function or the inverse of the loss weight for PDE residual 
        resi_sum_array: the PDE residual (summed over all temporal points)
        p1_prior, p2_prior, p3_prior: prior density function for each parameter
    
    Returns:
        KL divergence between the parameter prior and the likelihood function defined by 
        the exponential of the PDE residual    

    """
    q_vals = np.exp(-1*((1/(2*variance_array))*resi_sum_array) ) 

    partition_function = np.sum(q_vals)

    q_vals = q_vals/(partition_function+1e-8)

    prior_pdf = p1_prior[:, None, None] * p2_prior[None, :, None] * p3_prior[None, None, :] 

    partition_function =  np.sum(prior_pdf)

    prior_pdf = prior_pdf/partition_function

    return np.sum(prior_pdf * np.log(prior_pdf / (q_vals+1e-8)))



def three_marginal_prior(prob_grid, x0_vals, x1_vals, x2_vals):
    """
    Computes the marginal distribution for each parameter after integrating out others in prob_grid.
    This function is only used for visualizing projections of the distribution f (eqn.9).

    Args: (NumPy arrays)
        prob_grid :  distribution obtained after evaluating deviation between numerical solution and initial model
        x0_vals, x1_vals, x2_vals : the parameters' domains

    Returns: 
        the marginal distribution for each parameter descending from prob_grid
    """

    dx0 = x0_vals[2] - x0_vals[1]
    dx1 = x1_vals[2] - x1_vals[1]
    dx2 = x2_vals[2] - x2_vals[1]

    prob_grid /= (prob_grid.sum()*dx0*dx1*dx2)

    x0_marginal = np.sum(prob_grid, axis = (1,2))*dx1*dx2
    x1_marginal = np.sum(prob_grid, axis = (0,2))*dx0*dx2
    x2_marginal = np.sum(prob_grid, axis = (0,1))*dx0*dx1

    return x0_marginal, x1_marginal, x2_marginal

    

def three_stats_prior(prob_grid, x0_vals, x1_vals, x2_vals):
    """
    Computes the mode and standard deviation for each parameter from prob_grid -- 
    we take these quantities as characterizing the highest density region and use them
    to construct priors for the parameters.

    Args: (NumPy arrays)
        prob_grid: distribution obtained after evaluating deviation between numerical solution and initial model
        x0_vals, x1_vals, x2_vals : the parameters' domains


    Returns:
        the mode and standard deviation for each parameter, with the latter being
        = IQR/1.349 of each marginal distribution 
    
    """
    dx0 = x0_vals[2] - x0_vals[1]
    dx1 = x1_vals[2] - x1_vals[1]
    dx2 = x2_vals[2] - x2_vals[1]

    prob_grid /= (prob_grid.sum()*dx0*dx1*dx2)

    x0_marginal = np.sum(prob_grid, axis = (1,2))*dx1*dx2
    x1_marginal = np.sum(prob_grid, axis = (0,2))*dx0*dx2
    x2_marginal = np.sum(prob_grid, axis = (0,1))*dx0*dx1

    x0_cdf = np.zeros(len(x0_vals))
    norm_x0 = np.sum(x0_marginal)*dx0
    for i in range(len(x0_vals)):
      x0_cdf[i] = np.sum(x0_marginal[:i])*dx0/norm_x0

    x1_cdf = np.zeros(len(x1_vals))
    norm_x1 = np.sum(x1_marginal)*dx1
    for i in range(len(x1_vals)):
      x1_cdf[i] = np.sum(x1_marginal[:i])*dx1/norm_x1

    x2_cdf = np.zeros(len(x2_vals))
    norm_x2 = np.sum(x2_marginal)*dx2
    for i in range(len(x2_vals)):
      x2_cdf[i] = np.sum(x2_marginal[:i])*dx2/norm_x2

    flat_index = np.argmax(prob_grid)           
    mode_index = np.unravel_index(flat_index, prob_grid.shape) 
    
    x0_mode = x0_vals[mode_index[0]]
    x1_mode = x1_vals[mode_index[1]]
    x2_mode = x2_vals[mode_index[2]]

    Q1 = np.interp(0.25, x0_cdf, x0_vals)
    Q3 = np.interp(0.75, x0_cdf, x0_vals)
    x0_std = (Q3 - Q1)/1.349

    Q1 = np.interp(0.25, x1_cdf, x1_vals)
    Q3 = np.interp(0.75, x1_cdf, x1_vals)
    x1_std = (Q3 - Q1)/1.349

    Q1 = np.interp(0.25, x2_cdf, x2_vals)
    Q3 = np.interp(0.75, x2_cdf, x2_vals)
    x2_std = (Q3 - Q1)/1.349 

    return x0_mode, x1_mode, x2_mode, x0_std, x1_std, x2_std


@njit
def njit_marginal_ll(variance_R, resi_sum_array):
    """
    Computes the unnormalized likelihood function defined as the exponential of PDE residual 

    Args: (NumPy arrays)
        variance_R: variance parameter in the PDE residual (inverse of PDE residual loss weight)
        resi_sum_array: the PDE residual sum 

    Returns: 
        the unnormalized likelihood function 

    """
    pdf_unnorm = np.exp(-1*((1/(2*variance_R))*resi_sum_array) ) 

    return pdf_unnorm



@njit
def posterior_array_3D(obj, x0_prior, x1_prior, x2_prior, x0_vals, x1_vals, x2_vals):
    """
    Computes the posterior distribution given the likelihood function and prior

    Args: (NumPy arrays)
        obj: likelihood function
        x0_prior, x1_prior, x2_prior: prior functions for each parameter
        x0_vals, x1_vals, x2_vals: parameters' domains
    
    Returns: 
        posterior distribution 
    """
    posterior_init = np.empty((len(x0_vals), len(x1_vals), len(x2_vals)))
    for i in range(len(x0_vals)):
      for j in range(len(x1_vals)):
        for k in range(len(x2_vals)):
            posterior_init[i,j,k] = obj[i,j,k]*x0_prior[i]*x1_prior[j]*x2_prior[k]
    return posterior_init



def mean_var_KL_3D(variance_bound, likelihood_function, x0_prior, x1_prior, x2_prior):
    """
    Computes the optimal residual variance parameter that minimizes KL divergence between
    the likelihood function and the prior density defined by the product of three univariate Gaussian priors.
    This function is used towards deriving the hyperparameters of the PDE residual prior which we take to be an inverse-gamma function.
    
    Args: (NumPy arrays)
        variance_bound: initial variance value in the optimization 
        likelihood function: likelihood function derived from the PDE residual
        x0_prior, x1_prior, x2_prior: we take product of these univariate priors to be the prior density function
    
    Returns:
        (i) the PDE residual variance parameter that minimizes the KL divergence between the likelihood and the prior
        (ii) a graph to visually check that the minimization procedure has worked well. 
    
    """
    def objective(variance_array):
        vari = variance_array[0]
        return kl_divergence_3D(vari, likelihood_function, x0_prior, x1_prior, x2_prior)

    optimal_beta = minimize(objective, x0=[1.5*variance_bound], bounds=[(variance_bound, None)],method='Powell')

    print(f"Optimal beta: {optimal_beta.x[0]:.4e}")
    mean_var = optimal_beta.x[0]
    
    variance_vec = np.linspace(0.8*mean_var,1.2*mean_var,40)
    kl_list = []

    for var in variance_vec:
      kl_list.append(kl_divergence_3D(var, likelihood_function, x0_prior, x1_prior, x2_prior))

    plt.figure(figsize=(4,3))
    plt.plot(variance_vec, kl_list)
    plt.xlabel("Variance R")
    plt.ylabel("KL Divergence")
    plt.title('KL Divergence between prior and initial likelihood',fontsize = 8)

    return optimal_beta.x[0]



def PDE_prior(min_var, mean_var):
    """
    Computes the alpha and beta parameters of the inverse-gamma distribution assumed to be 
    the PDE residual prior.

    Args:(floats)
        min_var: the minimum variance as constrained by the resolution of the parameters' domains
        mean_var: the optimal variance that minimizes KL divergence between parameter prior and initial likelihood.

    Returns: 
        the alpha and beta parameters of the inverse-gamma prior for the PDE residual term

    """
    print(f'Mean_var: {mean_var:.3f}, Min_var: {min_var:.3f}')
    eps_s = 2*min_var/(mean_var - min_var)
    alpha_s = 1 + eps_s
    beta_s = eps_s*mean_var
    print(f'Alpha_s: {alpha_s:.4f}, Beta_s: {beta_s:.4f}')

    return alpha_s, beta_s
