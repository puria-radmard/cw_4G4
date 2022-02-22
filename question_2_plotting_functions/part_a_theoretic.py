import pylab
import matplotlib.cm as cm
import copy
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')

# For theoretical error calculations, to be compared to the Hopfield Network shortlty

# NB: all the below use statistics derived for 8H/N, which can easily be scaled 
# in downstream tasks

def mean_exact(N, M):
    'Exact" first moment of the CLT approximation of 8H/N'
    return (N-1)/(N)


def variance_exact(N, M):
    'Exact" second moment of the CLT approximation of 8H/N'
    numerator = 2 * (M-1) * (N-1) + 4
    denominator = N**2
    return numerator/denominator


def mean_approx(N, M):
    'Large N, M approximation of first moment of the CLT approximation of 8H/N'
    return 1


def variance_approx(N, M):
    'Large N, M approximation of second moment of the CLT approximation of 8H/N'
    return 2*M/N


def p_error_exact(N, M):
    '''
        Probability of incorrect sign, assuming the "exact" distribution statistics
        given above. i.e. p(r < 0) = CDF(- mean / std)

        Should tend to p_error_approx as N, M get large
    '''
    mean, variance = mean_exact(N, M), variance_exact(N, M)
    std = np.sqrt(variance)
    return norm.cdf(- mean / std)


def p_error_approx(N, M):
    '''
        Probability of incorrect sign, assuming the large N, M approximation distribution statistics
        given above. i.e. p(r < 0) = CDF(- mean / std)
    '''
    mean, variance = mean_approx(N, M), variance_approx(N, M)
    std = np.sqrt(variance)
    return norm.cdf(- mean / std)


def plot_fixed_M_p_error(Ms, Ns, axs, colormap, include_exact=True, include_approx=True):
    '''
        Given a set of values for M and N, this will calculate and plot lines of p_error for fixed M
        i.e.:
            x-axis = N changing
            y-axis = p_error
            lines = different M value

        Interpretation: for a fixed memory set requirement, how big should the network be to generate
        reliable dynamics using the covariance rule
    '''

    # Iterate over values of M, but also colors, as we would like approx and exact line values
    # to have the same color
    for c, M in zip(colormap, Ms):

        # Get the error lines specified for all values of N, for this value of M
        if include_approx:
            line = p_error_approx(Ns, M)

            # Approximate line will be dashdot texture
            axs.plot(Ns, line, linestyle = '-.', color=c, label = str(M))

        if include_exact:
            line = p_error_approx(Ns, M)
            
            # Only want to label this one if we don't already have a label for this value of M
            # (color of lines will be the same)
            approx_label = str(M) if not include_approx else None

            # Exact line will be block texture
            axs.plot(Ns, line, linestyle = '-', color=c, label = approx_label)

        axs.legend(title='Value of M')
        axs.set_xlabel('N')
        axs.set_ylabel('Probability of bit flip error')


def plot_fixed_N_p_error(Ms, Ns, axs, colormap, include_exact=True, include_approx=True):
    '''
        Given a set of values for M and N, this will calculate and plot lines of p_error for fixed N
        i.e.:
            x-axis = M changing
            y-axis = p_error
            lines = different N value

        Interpretation: for a fixed neural network size, how well can I store this many memories?
    '''

    # Iterate over values of N, but also colors, as we would like approx and exact line values
    # to have the same color
    for c, N in zip(colormap, Ns):

        # Get the error lines specified for all values of M, for this value of N
        if include_approx:
            line = p_error_approx(N, Ms)

            # Approximate line will be dashdot texture
            axs.plot(Ms, line, linestyle = '-.', color=c, label = str(N))

        if include_exact:
            line = p_error_approx(N, Ms)
            
            # Only want to label this one if we don't already have a label for this value of M
            # (color of lines will be the same)
            approx_label = str(N) if not include_approx else None

            # Exact line will be blocked texture
            axs.plot(Ms, line, linestyle = '-', color=c, label = approx_label)

        axs.legend(title='Value of N')
        axs.set_xlabel('M')
        axs.set_ylabel('Probability of bit flip error')


def exact_p_error_threshold_plot(Ms, Ns, axes, colormap, num_thresholds = 10):
    """
        x-axis: N
        y-axis: M
        line: M for that N which first gets lower than 1/N, 
            i.e. maximum number of memories we can store with p < eps for that network size
    """
    grid_Ms, grid_Ns = np.meshgrid(Ms, Ns)

    # p_errors[n, m] gives p_e for n neurons storing m memories
    p_errors = p_error_exact(grid_Ns, grid_Ms)

    # Roughly number of errors
    for alpha in range(1, 1 + num_thresholds):

        # Start getting the maximum M for each N
        line_Ms, line_Ns = [], []

        # For each value of N (along x-axis), get the value of M which is out operating point
        for i, (N, errors_per_M) in enumerate(zip(Ns, p_errors)):

            # Get the argument that comes closest to our adaptive limits
            arg_M = np.argmin(np.abs(errors_per_M - 2*alpha/N))
            closest_M = Ms[arg_M]

            line_Ms.append(closest_M)
            line_Ns.append(N)

        # Plot these Ms
        axes.plot(line_Ns, line_Ms, c=colormap[alpha-1], label=str(f'p(E) < {2*alpha}/N'))

    # plt.plot(Ns, 0.054*Ns, color = 'black', linestyle='--')

    # Now, do a softer plot of the normalised bit error rate
    # plt.pcolormesh(Ms, Ns, p_errors)

    axes.legend()
    axes.set_ylabel('M')
    axes.set_xlabel('N')


class ColorMap:
    "Indexable plt color map, used in the functions above"

    def __init__(self, num_colors:int, pallete:str):
        self.num_colors = num_colors
        
        cm = pylab.get_cmap(pallete)
        self.colours = [cm(1.* i / num_colors) for i in range(num_colors)]

    def __len__(self):
        return self.num_colors

    def __getitem__(self, i):
        return self.colours[i]


if __name__ == '__main__':

    fig, axs = plt.subplots(1, 3, figsize = (16, 6))

    # For the fixed M case, we want fewer Ms, but a long log range of Ns
    fixed_M_Ms = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] # This many memories
    fixed_M_Ns = np.logspace(1, 4, 500).round() # 10 to 10000 neurons (50 steps)
    colormap_M = ColorMap(len(fixed_M_Ms), 'gist_rainbow')

    plot_fixed_M_p_error(fixed_M_Ms, fixed_M_Ns, axs[0], colormap_M, include_exact=True, include_approx=False)

    # For the fixed N case, we want fewer Ms, but a long log range of Ns
    fixed_N_Ms = np.logspace(0, 5, 500).round() # 10 to 1000 memories (50 steps)
    fixed_N_Ns = [100, 200, 500, 1000, 2000, 5000, 10000] # This many neurons 
    colormap_N = ColorMap(len(fixed_N_Ns), 'gist_rainbow')
    plot_fixed_N_p_error(fixed_N_Ms, fixed_N_Ns, axs[1], colormap_N, include_exact=True, include_approx=False)

    # Finally, the full operating grid, the heaviest one
    num_thresholds = 20
    all_Ns = np.logspace(3, 4, 6000).round() # 100 to 100000 neurons
    all_Ms = np.logspace(1, 4, 5000).round() # 10 to 10000 memories
    colormap_eps = ColorMap(num_thresholds, 'gist_rainbow')
    exact_p_error_threshold_plot(all_Ms, all_Ns, axs[2], colormap_eps, num_thresholds)

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    # axs[2].set_xscale('log')

    plt.show()
