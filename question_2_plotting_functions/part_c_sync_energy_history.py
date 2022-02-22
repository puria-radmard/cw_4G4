from question_2_hopfield_network import *

def plot(N, Ms, axes, num_repeats=20):

    oscillation_results = []

    for M in tqdm(Ms):

        trials = []

        for _ in range(num_repeats):

            # initialise network
            network = BinaryHopfieldNetwork(N=N, M=M)

            # network.go_to_memory(4, 0)
            
            energy_in_updates = [network.get_energy()]

            for a in range(50):
                new_energy_in_updates = network.synchronously_update_neural_activity()
                energy_in_updates.append(network.get_energy())

            # energy_history = np.array(energy_in_updates)   
            energy_history = normalise_energy_history(np.array(energy_in_updates))

            trials.append(energy_history)

        mean_line = np.stack(trials).mean(0)
        if axes is not None:
            axes.plot(mean_line, label = M)

        oscillation_results.append(mean_line)

    if axes is not None:
        axes.legend()

    return oscillation_results



def plot_oscillation_profile(Ms, energy_hists, axes):

    oscill_points = []

    for energy_hist in energy_hists:

        oscill_points.append(abs(energy_hist[20::2] - energy_hist[19::2]).mean())

    axes.plot(Ms, oscill_points)


if __name__ == '__main__':

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))

    N = 100
    selected_Ms = [5, 50, 500, 5000, 50000]
    plot(N, selected_Ms, axs[0], num_repeats=20)

    all_Ms = np.logspace(1, 4, 50).round().astype(int)
    results = plot(N, all_Ms, None, num_repeats=200)
    plot_oscillation_profile(all_Ms, results, axs[1])

    # axs[1].set_xscale('log')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('Normalised energy')

    axs[1].set_xlabel('M')
    axs[1].set_ylabel('Average stable oscillation')

    axs[1].set_xscale('log')

    plt.show()
