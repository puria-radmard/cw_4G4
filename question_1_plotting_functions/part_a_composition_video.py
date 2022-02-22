from matplotlib import animation
from question_1_utils import *

# Select which images we want to show
image_ids = [5, 50, 500, 5000]

# Prepare canvas
fig, axes = plt.subplots(2, len(image_ids) + 1, figsize = (2 * (len(image_ids)) + 1, 4))

# Remove clutter and label
for ax in axes.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].set_ylabel('Images')
axes[1, 0].set_ylabel('Reconstructions')

# Plot the true photos on the top row
for img_id, ax in zip(image_ids, axes[0]):
    ax.imshow(show_image_by_idx(img_id, show = False)[0], cmap = 'gray')

# Get the decomposition that each of these has in latent space
latent_coeffs = get_latent_variable_by_image_idx(image_ids).numpy()

# Initialise the compositions
compositions = [np.zeros([image_side, image_side]) for _ in image_ids]

# Initialise video frames
frames = []

# Iterate through the latent variable coefficients as we add more onto them
for k in range(K):

    # Get the actual generative weights that this LV coeff correspond to
    wgen_k = show_wgen_by_idx(k, False).numpy()

    # This round of co-animations, starting with the updated reference filter
    k_imgs = [axes[-1, -1].imshow(wgen_k[0], cmap = 'gray')]

    # Iterate through the images and axes we are adding tp
    for (lcs, comp, ax) in zip(latent_coeffs, compositions, axes[1]):
        
        # Add the correct weighting of the generative weight to the composition
        comp += lcs[k] * wgen_k[0]

        # Plot progress to axes
        k_imgs.append(ax.imshow(comp, animated=True, cmap = 'gray'))

    # Add last frame to the video so far
    frames.append(k_imgs)

ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=10000)
ani.save('generative_model_movie.mp4')

plt.show()
