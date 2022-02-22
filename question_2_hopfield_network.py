import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import sys, os
from glob import glob

sns.set_style('darkgrid')


class BinaryHopfieldNetwork:
    """
        The default Hopfield network given in lecture, but with -1/+1 neural activations
        
        N = number of neurons in network
        M = number of memory patterns to store
        r = initialisation neural activities, of size [N, N] (optional)
        W = initialisation weights, of size [N, N] (optional, rare option)
    """
    
    def __init__(self, N, M, r=None, W=None):
        # Number of neurons in network
        self.N = N
        
        # Number of memory patterns to store
        self.M = M
        
        # See documentation for each function for details
        # Written like this to allow overwriting without rewriting init function
        self.init_memories()
        self.init_W(W)
        self.init_r(r)

    def init_memories(self):
        # Storing M memories, each using the full N neuron set
        # i.e. self.memories[m] gives the mth memory
        # self.memories[m, n] gives the nth neuron value for that memory
        self.memories = 2 * np.random.rand(self.M, self.N).round() - 1

    def init_W(self, W=None):
        
        # Initialise weights if not given
        # Commented out lines: previous implementation explicitly made sure
        # there was no autosynaptic connections. Now, I just remove the autosynaptic terms
        # from any calculations. See get_energy for an example
        if W != None:
            assert W.shape == (self.N, self.N), "Weight matrix of wrong size"
            # assert sum(np.diag(W) == 0.), "No autosynapsing allowed"
        else:
            W = self.apply_binary_covariance_rule()
            ## W -= np.diag(W)*np.eye(self.N)
            
        # NB: W[k,j] feed from j to k, see get_local_fields for an example
        self.W = W

    def init_r(self, r=None):
        # Initialise neural activities if not given
        if r != None:
            assert r.shape == self.N, "Neural activities vector of wrong size"
            assert all(np.in1d(r, [-1, 1])), "We are dealing with binary neurons"
        else:
            r = 2 * np.random.rand(self.N).round() - 1
        self.r = r

    def go_to_memory(self, idx, p_flip):
        # Initialise network at or near a memory, with the probability of bit flipping provided

        # This will be 1 at places we want to keep
        flip_mask = 2 * (np.random.rand(self.N) > p_flip).astype(int) - 1

        # Get the memory we are about to process
        uncorrupted_memory = self.memories[idx]

        # Corrupt the memory at place the network there
        self.r = flip_mask * uncorrupted_memory.copy()
        
    def apply_binary_covariance_rule(self):
        # Apply the covariance rule, based on the binary sequences self.memories,
        # to get the optimal weights for the Hopfield network
        # W_{ij} = \sum_{m=1}^M(r_i^{(m)}-\frac{1}{2})(r_j^{(m)}-\frac{1}{2})
        # W_{ii} = 0, although this is ignored (see __init__)
        
        # We are summing over the M size axis:
        W = (self.memories.T @ self.memories) / self.N

        # Remove autosynaptic connections
        W *= (np.ones([self.N, self.N]) - np.eye(self.N))
        return W

    def get_energy(self):
        # This is the energy function of the neurons, given the current weights
        # For the 0/1 neuron case, this is simply:
        # E(\bold{r}) = \frac{-1}{2}\sum_{i}\sum_{j\neq i}W_{ij}r_ir_j

        # So we have ot convert our -1/+1 case to that:
        zero_one_rs = 0.5 * (self.r + 1)
        
        full_inner_product = self.r.T @ self.W @ self.r
        autosynaptic_terms = (np.diag(self.W) * np.square(self.r)).sum()
        return -0.5 * (full_inner_product - autosynaptic_terms)
    
    def get_local_fields(self, *ks):
        # Get the local field:  H_k(t) = \sum_{j\neq k}W_{kj}r_j(t)
        # for the indices k provided. If ks not provided, return all local fields
        
        # If ks = [], return all values
        ks = ks if len(ks) else ...
        
        # Index the weights by their destination, i.e. do they feed into weights?
        all_weights_in = self.W[ks,:]   # size [len(ks), N]
        auto_synaptic_weights_in = np.diag(self.W)[ks]   # size [len(ks), ]
        
        # Get the total weighted activation into the relevant neurons
        H_incl_auto = all_weights_in @ (self.r + 1)  # size [len(ks), ]
        
        # Get the amount of this which is autosynaptic
        H_auto_contribution = auto_synaptic_weights_in * (self.r[ks] + 1)  # size [len(ks), ]
        
        # Total field is the difference of these
        return (self.N / 8) * (H_incl_auto - H_auto_contribution)  # size [len(ks), ]
    
    @staticmethod
    def nonlinearity(a):
        # Sign function
        return np.sign(a)
    
    def update_neural_activities_by_idx(self, *ks):
        # Update the neural activies (self.r) based on provided indices
        # If ks not provided, return all local fields

        # Get the raw inputs
        H_ks = self.get_local_fields(*ks) if len (ks) else self.get_local_fields()
        
        # Pass through nonlinearity
        new_activities = self.nonlinearity(H_ks)
        
        # Update the correct activities
        if ks:
            self.r[ks] = new_activities
        else:
            self.r[...] = new_activities
        
    def synchronously_update_neural_activity(self):
        # Update all the neural activities together. 
        # This is not in the lecture notes, but included for completeness
        self.update_neural_activities_by_idx()
        
    def asynchronously_update_neural_activity(self):
        # Update all the neural activities one by one
        # This is what the lecture notes do

        # Also, get the energy evolution over the course of the update.
        # This should be monotonoically non-increasing
        
        energy_history = [self.get_energy()]
    
        for i in range(self.N):
            self.update_neural_activities_by_idx(i)
            energy_history.append(self.get_energy())
        
        return np.array(energy_history)

    
class PictoralBinaryHopfieldNetwork(BinaryHopfieldNetwork):
    """
        Given a directory of images, this will pick then up and turn them
        into memories,
    """

    def __init__(self, directory, N, M):
        # Number of neurons in network
        self.N = N
        
        # Number of memory patterns to store
        self.M = M
        
        # Get the images from all txt files in the directory
        self.init_memories(directory)

        # Init weights and state like normal
        self.init_W()
        self.init_r()

    def get_memory_as_image(self, i):
        return vector_to_square(self.memories[i], self.N)

    def get_current_state_as_image(self, add_cursor=None):
        if add_cursor == None:
            return vector_to_square(self.r, self.N)
        else:
            # Adding a cursor would require RGB images, so we have to convert
            state_with_cursor = np.repeat(self.r[:, np.newaxis], 3, axis=1)
            state_with_cursor[add_cursor] = [1, 0, 0]
            return vector_to_square(state_with_cursor, self.N)

    def init_memories(self, directory):

        # Make sure we are given a directory
        assert os.path.isdir(directory)

        # Get all images in the directory
        image_paths = glob(os.path.join(directory, '*.txt'))

        # Initialise list of memories
        memories = []

        # Iterate through paths
        for image_path in image_paths:

            # Pick up this image and add it to the cache
            image_df = pd.read_csv(image_path, header=None, index_col=None, sep=',')
            image = image_df.values.astype(int)
            vectorised_image = square_to_vector(image, self.N)
            memories.append(vectorised_image)

        # Once all images are added to cache, numpyify and save to class
        self.memories = np.stack(memories)


def square_to_vector(img, N):
    # Reshape an N by N square of +1/-1 to a vector, also of +1/-1
    return img.reshape(N)


def vector_to_square(r, N):
    # Reshape a binary, +1/-1 vector into a 0/1 image of 1:1 aspect ratio

    # Turn -1/+1 image into a 0/1 one for imshow
    one_zero_vector = (0.5 * (r + 1)).astype(int)

    # Square image
    square_sides = int(np.sqrt(N))

    # Reshape image into square to get image
    if len(r.shape) == 1:
        # Greyscale
        image = one_zero_vector.reshape(square_sides, square_sides)
    else:
        # RGB
        one_zero_vector *= 255
        image = one_zero_vector.reshape(square_sides, square_sides, -1)

    return image


def normalise_energy_history(energy_history):
    # Normalise the energy drop to go from 1 to 0
    energy_history -= energy_history[-1]
    energy_history /= energy_history[0]
    return energy_history
