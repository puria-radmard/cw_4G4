from question_2_hopfield_network import *

def plot(N, Ms, axes, num_repeats = 20):

    for M in tqdm(Ms):

        trials = []

        for _ in range(num_repeats):

            # initialise network
            network = BinaryHopfieldNetwork(N=N, M=M)

            # network.go_to_memory(4, 0)
            
            energy_in_updates = []

            for a in range(5):
                new_energy_in_updates = network.asynchronously_update_neural_activity()
                energy_in_updates.extend(new_energy_in_updates)

            energy_history = np.array(energy_in_updates)   
            energy_history = normalise_energy_history(np.array(energy_in_updates))

            trials.append(energy_history)

        axes.plot(np.stack(trials).mean(0), label = M)


if __name__ =='__main__':

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))

    N = 100
    Ms = [1, 2, 5, 10, 50, 100, 200, 500, 1000, 20000]
    plot(N, Ms, axs[0], num_repeats = 20)

    N = 1000
    Ms = [1, 5, 10, 100, 500, 1000, 20000]
    plot(N, Ms, axs[1], num_repeats = 20)

    axs[0].legend()
    axs[1].legend()

    axs[0].set_xlabel('t')
    axs[1].set_xlabel('t')

    axs[0].set_ylabel('Normalised energy')

    plt.show()
