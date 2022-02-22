from question_2_hopfield_network import *
import matplotlib.animation as animation


def initialise_axes_for_hopfield(ps, nets, M):

    # Initialise video axes
    fig, axes = plt.subplots(M, len(ps) + 1, figsize = (2 * (len(ps) + 1), 2 * M))

    # Show original memory on leftmost column
    [axes[i, 0].imshow(nets[0, 0].get_memory_as_image(i)) for i in range(M)]

    # Label left most column
    axes[0, 0].set_title(f'Original memories (M = {M})')

    # Makes things easier downstream
    cut_axes = axes[:,1:]

    # Label level of corruption
    [ax.set_title(f'Initial corruption: $p = {p}$') for p, ax in zip(ps, cut_axes[0, :])]

    return fig, axes, cut_axes



all_ps = [0.01, 0.1, 0.2, 0.5, 0.9]

nets = np.array([
    PictoralBinaryHopfieldNetwork('question_2_states', 289, 2) for _ in range(2 * len(all_ps))
]).reshape(2, len(all_ps))


# Initialise each row at a corrupted memory
[net.go_to_memory(0, p) for p, net in zip(all_ps, nets[0])]
[net.go_to_memory(1, p) for p, net in zip(all_ps, nets[1])]

# Initialise video
frames = []

# Generate axes with original on LHS
fig, axes, cut_axes = initialise_axes_for_hopfield(all_ps, nets, 2)

# Remove clutter
for ax in axes.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])

# This many full asynchronous updates
for _ in range(1):

    # Rotate through neurons to update
    for i in tqdm(range(289)):#range(289):

        # Show the corrupted memory so far
        # Placed before update so we can see the original corruption too
        a = [ax.imshow(net.get_current_state_as_image(add_cursor=i), animated = True) for ax, net in zip(cut_axes[0], nets[0])]
        a.extend([ax.imshow(net.get_current_state_as_image(add_cursor=i), animated = True) for ax, net in zip(cut_axes[1], nets[1])])

        # Add all the frames to animate
        frames.append(a)

        # Update all networks - ONLY ONE PIXEL AT A TIME
        [[net.update_neural_activities_by_idx(i) for net in nets_row] for nets_row in nets]
        

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=10000)
ani.save('hopfield_movie.mp4')

plt.show()
