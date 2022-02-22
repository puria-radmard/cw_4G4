from question_1_utils import *

import seaborn as sns
sns.set_style('darkgrid')

# Get all the generative weights: (256, 32, 32)
reshaped_generative_weight = show_wgen_by_idx(..., None, False).numpy()

# Get all the FFTs of these generative weights
batch_fft = np.fft.fft2(reshaped_generative_weight)

# mean frequencies for each generative weight filter
# i.e. the mean magnitude of fft coeffs
mean_abs_freqs = abs(batch_fft.reshape(batch_fft.shape[0], -1).mean(1))

# Get ranking and corresponding frequency mags
top_kds = np.argsort(mean_abs_freqs)
sorted_freqs = mean_abs_freqs[top_kds]

# Which generative weights do we want to plot here?
sorted_chosen_ks = [10, 65, 110, 250, 254, 255]
# Exportable ks for later use
chosen_ks = [top_kds[pseudo_k] for pseudo_k in sorted_chosen_ks]

### EVERYTHING ABOVE THIS IS SHARED

if __name__ == '__main__':

    ## Systematic way of comparing frequencies
    fig, axs = plt.subplots(1, figsize=(8, 8))

    # Plot the range of these fft means
    axs.plot(sorted_freqs)

    # This whole bit has to be hard coded :(
    positions = [
        [-0.05,0.4,0.2,0.2],
        [0.10,0.08,0.2,0.2],
        [0.35,0.10,0.2,0.2],
        [0.68,0.09,0.2,0.2],
        [0.70,0.4,0.2,0.2],
        [0.72,0.8,0.2,0.2],
    ]
    coords = [
        ([scks], [sorted_freqs[scks]]) for scks in sorted_chosen_ks
    ]

    for k, pos, co in zip(chosen_ks, positions, coords):

        axs.scatter(*co, color = 'red')

        ins = axs.inset_axes(pos)
        ins.imshow(reshaped_generative_weight[k])
        ins.set_title(f'k = {k}')
        ins.set_xticks([])
        ins.set_yticks([])

    axs.set_xlabel('Sorted index')
    axs.set_ylabel('Mean FFT magnitude')

    plt.show()
