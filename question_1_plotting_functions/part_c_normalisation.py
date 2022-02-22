from question_1_utils import *

# Initialise our gain controller
gain_controller = GainController()

# Set up SGD with initial learning rage
init_lr = 0.1
optimizer = torch.optim.SGD(gain_controller.parameters(), init_lr)

gain_controller.train()

# Get latent variables to train the gain controller on
X = Y @ R
_, nlls = gain_controller.optimise(X, optimizer, max_steps=2)

# With the trained network get the variance we will use to normalise
gain_controller.eval()
conditional_variance = gain_controller.conditional_variance(X)

# Normalised representations
C = (X / torch.sqrt(conditional_variance)).detach().numpy()

# Start plotting specifics - a grid of unnormalised and a grid of normalised 2Ds

k1 = 241
k2 = 214

fig, axes = plt.subplots(3, 2)

show_wgen_by_idx(k1, axes[0,0])
show_wgen_by_idx(k2, axes[0,1])

plot_lv_histogram_by_compoent_idx(k1, k2, axes=axes[1,0])
plot_lv_histogram_by_compoent_idx(k1, k2, latent_arr=C, axes=axes[1,1])

axes[2, 0].plot(nlls)

plt.show()
