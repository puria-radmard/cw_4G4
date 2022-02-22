from question_1_utils import *


def make_circle_mask(r_outer, r_inner, square_size = 32):
    # This gets a mask that you can multiply patterns with to get a circular pattern
    # on a pixel grid. e.g. multiply with grating to get signal/mask from handout

    assert r_inner <= r_outer 

    # Central coordinate of grid
    center = square_size / 2 - 0.5

    # Initialise the pixel grid
    mask = np.zeros([square_size, square_size])

    # Iterate over all pixels
    for i in range(square_size):

        # Get vertical displacement from center
        y_out = i - center

        for j in range(square_size):

            # Get horizontal displacement from center
            x_out = j - center

            # Get euclidean distance from center
            euc_dist = (x_out**2 + y_out**2)**0.5

            # Decide if in range:
            if r_inner <= euc_dist <= r_outer:

                # If in range, make cell active
                mask[i,j] = 1

    return mask


def make_centered_grating(angle, contrast, freq=1.6, square_size=32):

    # Standard constraints
    assert -np.pi < angle <= np.pi  # redundant
    assert 0 <= contrast <= 1   # greyscale limits
    assert freq < 2 # Nyquist frequency

    # Central coordinate of grid
    center = square_size / 2 - 0.5

    # Get the normal to the sinusoid, using direction angle provided
    normal_vector = np.array([np.cos(angle), np.sin(angle)])

    # Initialise the pixel grid
    canvas = np.zeros([square_size, square_size])

    # Iterate over all pixels
    for i in range(square_size):

        # Get vertical displacement from center
        y_out = i - center

        for j in range(square_size):

            # Get horizontal displacement from center
            x_out = j - center

            # Get direction from center
            displacement_vector = np.array([x_out, y_out]).T

            # Get displacement along sinusoid
            t = displacement_vector @ normal_vector

            # Get the stimulus value
            # contrast argument describes difference in high and low values
            # i.e. double the magnitude
            canvas[i,j] = (0.5 * contrast * np.cos(2 * np.pi * t * freq)) + 0.5

    return canvas

INNER = 4

mask = make_centered_grating(np.pi/4, 0.3)
mask *= make_circle_mask(16, INNER)

signal = make_centered_grating(-np.pi/4, 1)
signal *= make_circle_mask(INNER, 0)

stimulus = signal + mask

plt.imshow(stimulus, cmap='gray')
plt.show()

