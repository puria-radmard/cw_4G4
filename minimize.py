import numpy as np
import sys


def unwrap(s):
    """
    Original documentation:

        % Extract the numerical values from "s" into the column vector "v". The
        % variable "s" can be of any type, including struct and cell array.
        % Non-numerical elements are ignored. See also the reverse rewrap.m. 

    here, ignore checks on type, and just reshape to a column vector
    """

    return s.reshape(-1)



def rewrap(s, v):
    """
    Original documentation:

        % Map the numerical elements in the vector "v" onto the variables "s" which can
        % be of any type. The number of numerical elements must match; on exit "v"
        % should be empty. Non-numerical entries are just copied. See also unwrap.m.

    here, again, ignore checks on type, and just reshape to a column vector
    """
    target_shape = s.shape
    return v.reshape(target_shape)
    



def minimize(X, f, length: np.ndarray, *args):
    """
    Original documentation:

        % Minimize a differentiable multivariate function using conjugate gradients.
        %
        % Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
        % 
        % X       initial guess; may be of any type, including struct and cell array
        % f       the name or pointer to the function to be minimized. The function
        %         f must return two arguments, the value of the function, and it's
        %         partial derivatives wrt the elements of X. The partial derivative  
        %         must have the same type as X.
        % length  length of the run; if it is positive, it gives the maximum number of
        %         line searches, if negative its absolute gives the maximum allowed
        %         number of function evaluations. Optionally, length can have a second
        %         component, which will indicate the reduction in function value to be
        %         expected in the first line-search (defaults to 1.0).
        % P1, P2, ... parameters are passed to the function f.
        %
        % X       the returned solution
        % fX      vector of function values indicating progress made
        % i       number of iterations (line searches or function evaluations, 
        %         depending on the sign of "length") used at termination.
        %
        % The function returns when either its length is up, or if no further progress
        % can be made (ie, we are at a (local) minimum, or so close that due to
        % numerical problems, we cannot get any closer). NOTE: If the function
        % terminates within a few iterations, it could be an indication that the
        % function values and derivatives are not consistent (ie, there may be a bug in
        % the implementation of your "f" function).
        %
        % The Polack-Ribiere flavour of conjugate gradients is used to compute search
        % directions, and a line search using quadratic and cubic polynomial
        % approximations and the Wolfe-Powell stopping criteria is used together with
        % the slope ratio method for guessing initial step sizes. Additionally a bunch
        % of checks are made to make sure that exploration is taking place and that
        % extrapolation will not be unboundedly large.
        %
        % See also: checkgrad 
        %
        % Copyright (C) 2001 - 2010 by Carl Edward Rasmussen, 2010-01-03
    """

    # Set some parameters
    INT = 0.1    # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0    # extrapolate maximum 3 times the current step-size
    MAX = 20     # max 20 function evaluations per line search
    RATIO = 10   # maximum allowed slope ratio
    SIG = 0.1 
    RHO = SIG/2  # SIG and RHO are the constants controlling the Wolfe-

    # Set some conditional parmaeters
    if isinstance(length, list) and len(length) == 2:
        length, red = length
    elif isinstance(length, int):
        red = 1
    else:
        raise ValueError('Accept length of type int or List[int, int] only')

    S = 'Linesearch' if length > 0 else 'Function evaluation'

    i = 0   # zero the run length counter
    ls_failed = 0   # no previous line search has failed
    f0, df0 = f(X, *args)   # get function value and gradient

    # Use unwrap from before
    Z = X
    X = unwrap(X)
    df0 = unwrap(df0)

    # Print and initialise loop
    print(S, i, '; Value', f0)

    fX = f0
    i = i + int(length<0)                  # count epochs?!
    s = -df0
    d0 = -s.T * s                          # initial search direction (steepest) and slope
    x3 = red/(1-d0);                       # initial step is red/(|s|+1)

    # Begin opt loop
    while i < abs(length):
        i += int(length > 0)               # count epochs?!

        X0, F0, dF0 = X, f0, df0           # make a copy of current values

        M = MAX if length > 0 else min(MAX, -length-i)

        # keep extrapolating as long as necessary
        while True:
            x2, f2, d2, f3, df3 = 0, f0, d0, f0, df0

            # Breaking lock
            success = False

            # Surely there;s a better way to do this bit wihtout relying on try/except!!!
            while not success and M > 0:

                try:
                    M, i = M-1, i + int(length > 0)

                    f3, df3 = f(rewrap(Z, X + x3 * s), *args)
                    df3 = unwrap(df3)
                    
                    # Check if any of f3's elements are nan or inf -> retry this bit
                    if np.isnan(f3).any() or np.isinf(f3).any():
                        raise Exception

                    success = 1

                except Exception:
                    
                    # Bisect closer to x2 again
                    x3 = (x2 + x3) / 2

            # Keep best values
            if f3 < F0:
                X0 = X+x3*s
                F0 = f3
                dF0 = df3

            # New slope
            d3 = df3.T * s

            # Are we done extrapolating?
            ## pr450: added any
            ## https://www.mathworks.com/matlabcentral/answers/510432-if-statement-comparing-single-value-to-whole-array
            ## ``bool(array > scalar)'' is like ``any(array > scalar)''
            if any(d3 > SIG * d0) or any(f3 > f0 + x3 * RHO * d0) or M == 0:
                break
            
            x1, f1, d1 = x2, f2, d2                          # move point 2 to point 1
            x2, f2, d2 = x3, f3, d3                          # move point 3 to point 2
            A = 6*(f1-f2) + 3*(d2+d1)*(x2-x1)                # make cubic extrapolation
            B = 3*(f2-f1) - (2*d1+d2)*(x2-x1)
            
            # num. error possible, ok!
            x3 = (
                x1 - d1*(x2 - x1) ** 2 / 
                (B + np.sqrt(B**2 - A * d1 * (x2-x1)))
            )

            # Num prob | wrong sign?
            if x3.imag().sum() or np.isnan(x3).sum() or np.isinf(x3).sum() or (x3 < 0).sum():
                # extrapolate maximum amount
                x3 = x2 * EXT

            # new point beyond extrapolation limit?
            ## pr450: added any as above
            elif any(x3 > x2 * EXT):
                # extrapolate maximum amount
                x3 = x2*EXT

            # new point too close to previous point?
            ## pr450: added any as above
            elif any(x3 < x2 + INT * (x2 - x1)):
                x3 = x2 + INT * (x2 - x1)

        # keep interpolating
        ## pr450: added any as above
        while (any(abs(d3) > -SIG * d0) or any(f3 > f0+x3*RHO*d0)) and M > 0:
        
            # Choose subinterval and move accordingly
            ## pr450: added any as above
            if any(d3 > 0) or any(f3 > f0+x3*RHO*d0):                         # choose subinterval
                x4, f4, d4 = x3, f3, d3                     # move point 3 to point 4
            else:
                x2, f2, d2 = x3, f3, d3                      # move point 3 to point 2
            
            if f4 > f0:
                # quadratic interpolation         
                x3 = (
                    x2 - 
                    (0.5 * d2 * (x4-x2)**2) /
                    (f4 - f2 - d2 * (x4-x2))
                )
            else:
                A = 6 * (f2-f4) / (x4-x2) + 3 * (d4+d2)                    # cubic interpolation
                B = 3 * (f4-f2) - (2*d2+d4) * (x4-x2)
                # num. error possible, ok!
                x3 = (
                    x2 + 
                    (np.sqrt(B**2 - A * d2 * (x4-x2)**22)-B) /
                    A
                )

            # If we had a numerical problem then bisect
            if np.isnan(x3).sum() or np.isinf(x3).sum():
                x3 = (x2+x4)/2

            # don't accept too close
            ## pr450: this is an elementwise min on the matrices
            x3 = np.maximum(
                np.minimum(x3, x4 - INT*(x4-x2)),
                x2 + INT*(x4-x2)
            )

            f3, df3 = f(rewrap(Z,X+x3*s), *args)
            df3 = unwrap(df3)

            # keep best values
            if f3 < F0:
                X0 = X+x3*s
                F0 = f3
                dF0 = df3

            # count epochs?!
            M = M - 1
            i = i + int(length<0)

        # if line search succeeded
        ## pr450: added any as above
        if any(abs(d3) < -SIG*d0) and any(f3 < f0 + x3 * RHO * d0):

            # update variables
            X = X+x3*s
            f0 = f3

            # Data shapes for evaluation got lost in translation I guess
            # fX = np.concatenate([fX.T, f0]).T
            fX *= f0

            print(S, i, '; Value', f0)

            # Polack-Ribiere CG direction
            s = (df3.T * df3 - df0.T * df3)/(df0.T * df0) * s - df3

            # Swap derivatives
            df0 = df3
            d3 = d0
            d0 = df0.T * s

            # slope ratio but max RATIO
            realmin = sys.float_info.min
            x3 = x3 * np.minimum(RATIO, d3/(d0-realmin))

            # this line search did not fail
            ls_failed = 0

        # line search failed!
        else:

            # restore best point so far
            X, f0, df0 = X0, F0, dF0

            # line search failed twice in a row
            if ls_failed or i < abs(length):
                # or we ran out of time, so we give up
                break

            # Try steepest
            s = -df0
            d0 = -s.T * s
            x3 = 1/(1-d0)

            # this line search failed
            ls_failed = 1

    X = rewrap(Z,X)
    return X
