import scipy.io
import time
import numpy as np
from torch import nn
import torch, math, sys
from minimize import minimize
import matplotlib.pyplot as plt
from scipy import optimize


# Pick up data from file
representational_data = scipy.io.loadmat('representational.mat')
datasets = [representational_data['Y'], representational_data['W'], representational_data['R']]
Y, W, R = list(map(lambda x: torch.tensor(x, dtype=torch.float64), datasets))

# Define constants
N, D = Y.shape
D, K = W.shape
assert (D, K) == R.shape

# Side lengths of image data and latent variable sides
image_side = int(D**0.5)
lv_side = int(K**0.5)


## Some plotting utils
def _show_image(img, side_size, show, axes):
    img = img.reshape(-1, side_size, side_size)
    if show:
        axes.imshow(img[0], cmap='gray')
    return img


def show_image_by_idx(n, axes, show = True):
    # Take the nth image in the dataset and show it
    img = Y[n, :]
    return _show_image(img, image_side, show, axes=axes)


def show_wff_by_idx(k, axes, show = True):
    # Take the kth inferential weight set and show it
    img = R[:, k]
    return _show_image(img, image_side, show, axes=axes)


def show_wgen_by_idx(k, axes, show = True):
    # Take the kth generative weight set and show it
    img = W[:, k].T
    return _show_image(img, image_side, show, axes=axes)


def get_latent_variable_by_image_idx(n, axes, show = True):
    # Show the latent variable grid for the nth image
    return Y[n] @ R


def show_generative_creation_by_idx(n, axes, show = True):
    # Show the recreation of the axes, nth image using the transpose generative weights
    latents = get_latent_variable_by_image_idx(n, False)
    img = latents @ W.T
    return _show_image(img, image_side, show, axes=axes)


def get_lv_dist_by_components(*ks):
    # ks is a list of indices, which are the component indies of the LVs (x) returned
    # i.e. this is for all the images, for histogramming purposes
    # if ks not provided, return all latents      
    return Y @ R[:, ks if len(ks) else ...]


def colour_scatter(x, y, num_bins, axes = plt):
    # plt.hist2d is not great for our case, so I will scatter and colour points based
    # local density

    #histogram definition
    bins = [num_bins, num_bins] # number of bins

    # histogram the data
    hh, locx, locy = np.histogram2d(x, y, bins=bins)
    
    # Sort the points by density, so that the densest points are plotted last
    z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]
    
    # For better viewing
    color = np.log(z2/z2.sum())

    axes.scatter(x2, y2, c=color, cmap='jet', marker='.')  


def plot_lv_histogram_by_compoent_idx(*ks, axes=plt, latent_arr=None):
    # Plot either the univariate histogram of x values at index k
    # or the 2D histogram of x values at indices k[0] and k[1]
    # If latent arr is not provided, use the standard values, i.e get X = Y @ R and index k

    NoneType = type(None)

    if not isinstance(latent_arr, NoneType):
        assert latent_arr.shape == (N, K)
    else:
        lvs = get_lv_dist_by_components(*ks).numpy()

    if len(ks) == 1:
        x_dist = lvs if isinstance(latent_arr, NoneType) else latent_arr[:,ks[0]]
        axes.hist(x_dist, 100)
    
    elif len(ks) == 2:
        x1_dist, x2_dist = lvs.T if isinstance(latent_arr, NoneType) else latent_arr[:,ks].T
        import pdb; pdb.set_trace()
        colour_scatter(x1_dist, x2_dist, 100, axes = axes)
    
    else:
        raise ValueError(f'Cannot histogram {len(ks)} different components (only 1 or 2)')


## Now the functions for Rasmussen's minimize function
def batch_conditional_variance(X, log_A, log_B, use_numpy=False):
    # Sigma matrix in report
    
    # Assert the sizes we are inputting, and that parameters have no negative elements
    assert X.shape == (N, K)
    assert log_A.shape == (K, K) 
    assert log_B.shape == (K,  ) 
    
    # For ease of implementation, we double up computation a little bit:
    # sigma_k**2 = - A[k,k] X[k]**2 + \sum_{all j} A[k,j] X[j]**2 + B[k]
    
    # Hence, the variance matrix of size (N, K) is:
    # variance = - X_squared @ diag(A) + X_squared @ A + B

    pkg = np if use_numpy else torch

    A = pkg.exp(log_A)
    B = pkg.exp(log_B)
    
    # For reuse - size [N, K]
    X_squared = pkg.square(X)
    assert X_squared.shape == (N, K)
    
    # Removed term - size [N, K]
    ## autoweights = pkg.outer(pkg.ones(N), pkg.diag(A))
    ## assert autoweights.shape == (N, K)
    ## autoterm = - X_squared * autoweights
    
    # Sum term - size [N, K]
    # Remove diagonal terms to get sum
    no_autosynaptic_weight_A = A * (pkg.ones_like(A) - pkg.eye(A.shape[0]))
    sum_term = X_squared @ no_autosynaptic_weight_A
    
    assert sum_term.shape == (N, K)#  and (no_autosynaptic_weight_A <= 0).sum() == A.shape[0]
    
    # Bias term
    bias_term = pkg.outer(pkg.ones(N), B)

    # Put it all together
    return sum_term + bias_term


def negative_batch_log_likelihood(X, log_A, log_B, use_numpy=False):
    # Compute the negative loglikelihood over a batch of images for a given parameter set
    # Sum of matrix L in report

    pkg = np if use_numpy else torch

    A = pkg.exp(log_A)
    B = pkg.exp(log_B)
    
    # Get the variance of size (N, K)
    batch_variance = batch_conditional_variance(X, A, B, use_numpy = use_numpy)

    # First, create a size (N, K) loglikelihood matrix, then we'll sum it all up
    # Normalisation term is in actual log
    norm_term = 0.5 * pkg.log(2 * math.pi * batch_variance)
    
    # Previously exponential term is now not in log
    exp_term = 0.5 * pkg.square(X) / batch_variance

    # Sum over N and K
    return (norm_term + exp_term).sum()


def d_negative_batch_log_likelihood(X, log_A, log_B, use_numpy=False):
    # Derivative of the above, given by:
    # dL_{n,k}/db_{k}   = ((1/sigma_k**2) - (x_{n,k}^2/sigma_k**4)) / 2
    # dL_{n,k}/da_{k,j} = x_{n,j} * dL_{n,k}/db_{k}

    pkg = np if use_numpy else torch

    A = pkg.exp(log_A)
    B = pkg.exp(log_B)
    
    # Variance vector, size (N, K)
    batch_variance = batch_conditional_variance(X, A, B, use_numpy = use_numpy)
    
    # Get the dL/db by data index and dimension index, size (N, K)
    # Hence, dL_db[n,k] = dL_{n,k}/db_{k}
    dL_db = 0.5 * ((1 / batch_variance) - (X**2 / batch_variance**2))
    assert dL_db.shape == (N, K)
    
    # We want: dL_dA[n,k,j] = dL_{n,k}/da_{k,j}
    # However, because storing and [N, K, K] tensor gets quite large,
    # we choose instead to add this directly.
    # i.e. we bypass the matrix stage and go straight to the summation over datapoints
    autosynaptic_dL_dA = 0
    for x_n, dL_db_row in zip(X, dL_db):
        autosynaptic_dL_dA += pkg.outer(x_n, dL_db_row)

    # Now remove the autosynaptic contribution with a simple matmul
    dL_dA = (pkg.ones([K, K]) - pkg.eye(K)) * autosynaptic_dL_dA
    
    # Sum dL_db over datapoints, as we have already done for dL_dA above
    return dL_dA, dL_db.sum(0)


def interface_function(log_theta, latents, _K = K, use_numpy=True, include_derivative=True):
    # My backend works with a [K, K] sized A matrix, a [K] sized B vector (and [N, K] sized latents)
    # The latents are fine like this, but the A matrix and B vector must be made into a 
    # column vector to use with minimize function
    
    # Hence, we take in latents and pass it onto other functions, but take
    # theta of size [K(K + 1), ] and reshape it into the A and B objects
    
    # NOTE: certain values of theta are effectively zero - these correspond to A[k,k]
    # i.e. theta[0], theta[K+1], ... are all zero

    # However, these are dealt with downstream. Namely, they are not included in the 
    # evaluation process, so even if they take on some crazy values due to optimisation,
    # they do not change the conditional variance model. The only problem
    # might be if they prevent the convergence criterion from being met.

    # Additionally, arithmetic is much quicker when things are off torch and on numpy, so we do that...

    assert log_theta.shape[0] == _K*(_K + 1)
    assert len(log_theta.shape) == 1
    log_A = log_theta[:_K*_K].reshape(_K, _K)
    log_B = log_theta[_K*_K:]

    # Scalar value returned, which can just be passed on
    negative_log_likelihood = negative_batch_log_likelihood(X=latents, log_A=log_A, log_B=log_B, use_numpy=use_numpy)

    # Return as numpy, so we can use Rasmussen's minimize function
    if include_derivative:
            
        # These have shapes:
        # raw_d_negative_log_likelihood_dA: [K, K]
        # raw_d_log_likelihood_dB: [K, ]
        raw_d_negative_log_likelihood_dA, raw_d_negative_log_likelihood_dB = d_negative_batch_log_likelihood(
            X=latents, log_A=log_A, log_B=log_B, use_numpy=use_numpy
        )
        
        # NOTE: we ``might'' have to use raw_d_negative_log_likelihood_dA.T, test it out!
        d_negative_log_likelihood_dA = raw_d_negative_log_likelihood_dA.reshape(-1)
        d_negative_log_likelihood = np.concatenate([d_negative_log_likelihood_dA, raw_d_negative_log_likelihood_dB])

        return negative_log_likelihood, d_negative_log_likelihood

    else:
        return negative_log_likelihood


## Now the pytorch interface
class GainController(nn.Module):

    """
        PyTorch wrapper for the conditional variance gain controller.

        Apply gain control by calling self, i.e. using self(X)
        Access the conditional variance directly using self.conditional_variance(X)
        Acts as its own loss function by calling self.nll()

        Dont require d_negative_batch_log_likelihood thanks to autograd

        Can pass A and B matrices in
    """

    def __init__(self, log_A_init=None, B_init=None, _K=K):
        super(GainController, self).__init__()

        assert True if log_A_init == None else (log_A_init.shape == (_K, _K))
        assert True if B_init == None else (B_init.shape == (_K,  ))
        
        # Store latent variable size
        self.K = K

        # Standard parameters, which will require autograd
        self.log_A = nn.Parameter(torch.randn(self.K, self.K) if log_A_init == None else log_A_init)
        self.log_B = nn.Parameter(torch.randn(self.K,  ) if B_init == None else B_init)

        self.double()

    def conditional_variance(self, X):
        # Get the conditional variance matrix, of size [N, K]
        return batch_conditional_variance(X, self.log_A, self.log_B)

    def nll(self, X):
        # Get the negative loglikelihood of current parmeters
        return negative_batch_log_likelihood(X, self.log_A, self.log_B)

    def optimise(self, latents, optimizer, scheduler = None, convergence_threshold = 0.05, max_steps = 1000):

        # convergence based on (quadratic) closness of consecutive loglikelihoods
        prev_nll = float('inf')
        t = 0

        # Set start time and initialise log arrays
        start_time = time.time()
        nlls = []
        times = []

        while t < max_steps:

            t += 1

            # Reset optimizer
            optimizer.zero_grad()

            # Get the loss function we want to minimize
            nll = self.nll(latents)

            # Apply loss function update
            nll.backward()
            optimizer.step()

            print(f'Step {t} || NLL {nll}')

            if torch.square(nll - prev_nll) < convergence_threshold:
                break
            
            # update_log and convergence requirement
            nlls.append(nll)
            times.append(time.time() - start_time)
            prev_nll = nll

        return times, nlls




if __name__ == '__main__':  

    latents = Y @ R
    init_log_theta = torch.randn(K * (K + 1), dtype=torch.float64)

    if sys.argv[1] == 'funcs':

        nllh_numpy, dnllh_numpy = interface_function(init_log_theta.numpy(), latents.numpy(), use_numpy=True)
        print('Numpy implementation:', nllh_numpy, dnllh_numpy)

        nllh_torch, dnllh_torch = interface_function(init_log_theta, latents, use_numpy=False)
        print('Torch implementation:', nllh_torch, dnllh_torch)

    elif sys.argv[1] == 'numpy_minimise_nll':

        # Use numpy explicit in this
        print(init_log_theta.numpy())
        final_theta = minimize(init_log_theta.numpy(), interface_function, 15, latents.numpy(), )
        print(final_theta)

    elif sys.argv[1] == 'scipy_minimise_nll':

        # Use numpy explicit in this
        print(init_log_theta.numpy())
        
        final_theta = optimize.minimize(
            interface_function, init_log_theta.numpy(),
            args = (latents, K, True, False)
        )
        print(final_theta)

    elif sys.argv[1] == 'torch_minimise_nll':

        # Nottpassing our premade parameters to this, as the breaking/reshaping is done automatically in the class
        gain_controller = GainController()

        # Set up SGD with initial learning rage
        init_lr = float(sys.argv[2])
        opt_type = eval(f'torch.optim.{sys.argv[3]}')
        optimizer = opt_type(gain_controller.parameters(), init_lr)

        times, nlls = gain_controller.optimise(latents, optimizer)
        
        # Save config and result to file
        log_dict = {
            'init_lr': init_lr,
            'opt_type': sys.argv[3],
            'scheduler': None,
            'times': times,
            'nlls': nlls,
            'model': gain_controller
        }
        torch.save(log_dict, f'optim_results/results_{init_lr}_{sys.argv[3]}_{time.time()}', 'w')
