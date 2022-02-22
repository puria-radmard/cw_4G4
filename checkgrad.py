import numpy as np

def checkgrad(f: callable, X: np.ndarray, e: float, *Ps):
    """
    Original MaatLab documentation:

        % checkgrad checks the derivatives in a function, by comparing them to finite
        % differences approximations. The partial derivatives and the approximation
        % are printed and the norm of the diffrence divided by the norm of the sum is
        % returned as an indication of accuracy.
        %
        % usage: checkgrad('f', X, e, P1, P2, ...)
        %
        % where X is the argument and e is the small perturbation used for the finite
        % differences. and the P1, P2, ... are optional additional parameters which
        % get passed to f. The function f should be of the type 
        %
        % [fX, dfX] = f(X, P1, P2, ...)
        %
        % where fX is the function value and dfX is a vector of partial derivatives.
        %
        % Carl Edward Rasmussen, 2001-08-01.

    pythonised 2022 pr450
    """

    # Original arguments went P1 ... P14, so we just keep fidelty
    assert len(Ps) <= 14, 'Only allowed <=14 extra arguments'

    # Evaluate the function and its partial derivative for the current value
    y, dy = f(X, *Ps)

    # Initialise pertubation evaluation array, of shape [dim(X), 1]
    dh = np.zeros_like(X).reshape(-1, 1)

    # Iterating over elements of X
    for j in range(len(X)):

        # Perturb argument in one dimension and evaluate
        dx = np.zeros_like(X).reshape(-1, 1)
        dx[j] += e
        y2, dy2 = f(X + dx, *Ps)

        # Perturb argument in negative one dimension and evaluate
        y1, dy1 = f(X - dx, *Ps)

        # Add result to main perturbation array
        dh[j] = (y2 - y1)/(2 * e)

    # Print the two vectors
    print(dy, dh)

    # Return norm of the pd difference divided by norm of the pd sum
    return np.norm(dh-dy)/np.norm(dh+dy)
