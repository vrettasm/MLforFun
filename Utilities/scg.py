# Generic imports.
import numpy as np
from ml_util import numerical_derivative

# Public interface.
__all__ = ['SCG']

# Current version.
__version__ = '0.0.1'

# Author.
__author__ = "Michalis Vrettas, PhD - Email: michail.vrettas@gmail.com"


# SCG optimization class definition.
class SCG(object):
    """
    Description:
    This class implements a Scaled Conjugate Gradient optimization object.

    NOTE: This code is adopted from NETLAB (a free MATLAB library).

    Reference Book:
    (1) Ian T. Nabney (2001): Netlab: Algorithms for Pattern Recognition.
        Advances in Pattern Recognition, Springer. ISBN: 1-85233-440-1.
    """

    def __init__(self, nit=500, tol_x=1.0E-6, tol_fx=1.0E-8, update=100, diagnostics_on=False):
        """
        Description:
        Constructor for an SCG object.

        Args:
        - nit (int): maximum number of iterations.
        - tol_x (float): precision in 'x' (input) space.
        - tol_fx (float): precision in 'f(x)' (output) space.
        - update (int): frequency to update the output and store the statistics.
        - diagnostics_on (bool): flag that signals the monitoring of the algorithm stats.

        Raises:
        - ValueError: if some input parameter is out of range.
        - TypeError: if some input parameter has the wrong type.
        """

        # Check the input for correct type.
        if isinstance(nit, int):
            # Check the input for correct range.
            if nit > 0:
                # Assing the value.
                self._nit = nit
            else:
                raise ValueError(" SCG: Number of max iterations must be > 0.")
        else:
            raise TypeError(" SCG: Number of max iterations must be integer.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(tol_x, float):
            # Check the input for correct range.
            if tol_x > 0.0:
                # Assing the value.
                self._tol_x = tol_x
            else:
                raise ValueError(" SCG: Precision in 'x' must be > 0.")
        else:
            raise TypeError(" SCG: Precision in 'x' must be float.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(tol_fx, float):
            # Check the input for correct range.
            if tol_fx > 0.0:
                # Assing the value.
                self._tol_fx = tol_fx
            else:
                raise ValueError(" SCG: Precision in 'fx' must be > 0.")
        else:
            raise TypeError(" SCG: Precision in 'fx' must be float.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(update, int):
            # Check the input for correct range.
            if update > 0:
                # Assing the value.
                self._upd = update
            else:
                raise ValueError(" SCG: Number of update must be > 0.")
        else:
            raise TypeError(" SCG: Number of update must be integer.")
        # _end_if_

        # Check the input for correct type.
        if type(diagnostics_on) is bool:
            # Assing the value.
            self._diagnostics_on = diagnostics_on
        else:
            raise TypeError(" SCG: Diagnostics flag must be bool.")
        # _end_if_

    # _end_def_

    @property
    def nit(self):
        """
        Description:
        Accessor (get) for the maximum number of iterations.
        """
        return self._nit

    # _end_def_

    @property
    def tol_x(self):
        """
        Description:
        Accessor (get) for the precision in input space 'x'.
        """
        return self._tol_x

    # _end_def_

    @property
    def tol_fx(self):
        """
        Description:
        Accessor (get) for the precision in output space 'f(x)'.
        """
        return self._tol_fx

    # _end_def_

    @property
    def update(self):
        """
        Description:
        Accessor (get) for the update frequency.
        """
        return self._upd

    # _end_def_

    @property
    def diagnostics_on(self):
        """
        Description:
        Accessor (get) for the update.
        """
        return self._diagnostics_on

    # _end_def_

    @nit.setter
    def nit(self, n):
        """
        Description:
        Accessor (set) for the maximum number of iterations.
        """
        # Type check.
        if isinstance(n, int):
            # Range check.
            if n > 0:
                self._nit = n
            else:
                raise ValueError(" SCG: Number of max iterations must be > 0.")
            # _end_if_
        else:
            raise TypeError(" SCG: Type of max iterations must be int.")
        # _end_if_

    # _end_def_

    @tol_x.setter
    def tol_x(self, n):
        """
        Description:
        Accessor (set) for the precision in 'x'.
        """
        # Type check.
        if isinstance(n, float):
            # Range check.
            if n > 0.0:
                self._tol_x = n
            else:
                raise ValueError(" SCG: Precision for convergence in 'x' must be > 0.")
            # _end_if_
        else:
            raise TypeError(" SCG: Precision in 'x' must be of type float.")
        # _end_if_

    # _end_def_

    @tol_fx.setter
    def tol_fx(self, n):
        """
        Description:
        Accessor (set) for the precision in 'f(x)'.
        """
        # Type check.
        if isinstance(n, float):
            # Range check.
            if n > 0.0:
                self._tol_fx = n
            else:
                raise ValueError(" SCG: Precision for convergence in 'f(x)' must be > 0.")
            # _end_if_
        else:
            raise TypeError(" SCG: Precision in 'f(x)' must be of type float.")
        # _end_if_

    # _end_def_

    @update.setter
    def update(self, n):
        """
        Description:
        Accessor (set) for the update frequency.
        """
        # Type check.
        if isinstance(n, int):
            # Range check.
            if n > 0:
                self._upd = n
            else:
                raise ValueError(" SCG: Update frequency must be > 0.")
            # _end_if_
        else:
            raise TypeError(" SCG: Type of update frequency must be int.")
        # _end_if_

    # _end_def_

    @diagnostics_on.setter
    def diagnostics_on(self, n):
        """
        Description:
        Accessor (set) for the diagnostics flag.
        """
        # Type check.
        if type(n) is bool:
            self._diagnostics_on = n
        else:
            raise TypeError(" SCG: Type of diagnostics flag is bool.")
        # _end_if_

    # _end_def_

    def optimize(self, f, x0, df=None, *args):
        """
        Description:
            Scaled conjugate gradient optimization, attempts to find a local minimum
            of the function f(x). Here 'x0' is a column vector and 'f' returns a scalar
            value. The minimisation process uses also the gradient 'df' (i.e. df(x)/dx).
            The point at which 'f' has a local minimum is returned as 'x'. The function
            value at that point (the minimum) is returned in "fx".

        Input parameters:
            f      : is the objective function to be optimised.
            x0     : is the initial point of function (D x 1).
            df     : is the derivative of the objective function w.r.t 'x'.
            *args  : additional parameters for both 'f' and 'df' functions.

        Output parameters:
            x_opt  : the point where the minimum was found.
            f_opt  : the function value, at the minimum point.
            stat   : statistics that collected through the optimisation process.
        """

        # Display method name.
        print(" >> SCG optimization ")

        # Ensure compatibility.
        x0 = np.asfarray(x0).flatten()

        # Dimensionality of the input vector.
        D = x0.size

        # Statistics dictionary.
        stat = {'Itr': self._nit, 'f_eval': 0, 'g_eval': 0}

        # If the diagnostics is on we add more fields.
        if self._diagnostics_on:
            stat['Fx'] = np.zeros(self._nit)
            stat['Gx'] = np.zeros(self._nit)
            stat['beta'] = np.zeros(self._nit)
        # _end_if_

        # Initialization.
        x = np.asfarray(x0).copy()

        # Initial sigma value.
        sigma0 = 1.0E-4

        # Initial function/gradients value.
        fnow = f(x, *args)

        # If a derivative function has been given use it.
        if df:
            gradnew = df(x, *args)
        else:
            gradnew = numerical_derivative(f, x, method='cdf', *args)
        # _end_if_

        # Increase function/gradient evaluations by one.
        stat['f_eval'] += 1
        stat['g_eval'] += 1

        # Store the current values.
        fold = fnow

        # Store the current gradient.
        gradold = gradnew

        # Setup the initial search direction.
        d = -gradnew

        # Force calculation of directional derivatives.
        success = True

        # Counts the number of successes.
        nsuccess = 0

        # Initial scale parameter.
        beta = 1.0

        # Lower & Upper bounds on scale (beta).
        betaMin = 1.0E-15
        betaMax = 1.0E100

        # Get the machine precision constant.
        eps_float = np.finfo(float).eps

        # Main optimization loop.
        for j in range(self._nit):
            # Calculate first and second directional derivatives.
            if success:
                # ...
                mu = d.dot(gradnew)
                if (mu >= 0.0):
                    d = -gradnew
                    mu = d.dot(gradnew)
                # _end_if_

                # Compute kappa and check for termination.
                kappa = d.dot(d)
                if (kappa < eps_float):
                    fx = fnow;
                    stat['Itr'] = j
                    return {"x_opt": x, "f_opt": fx, "stat": stat}
                # _end_if_

                # Update sigma and check the gradient on a new direction.
                sigma = sigma0 / np.sqrt(kappa)
                xplus = x + sigma * d

                # We evaluate only df(xplus).
                if df:
                    gplus = df(xplus, *args)
                else:
                    gplus = numerical_derivative(f, xplus, method='cdf', *args)
                # _end_if_

                # Increase function/gradients evaluations by one.
                stat['g_eval'] += 1

                # Compute theta.
                theta = (d.dot(gplus - gradnew)) / sigma
            # _end_if_

            # Increase effective curvature and evaluate step size alpha.
            delta = theta + beta * kappa
            if (delta <= 0.0):
                delta = beta * kappa
                beta = beta - (theta / kappa)
            # _end_if_
            alpha = -(mu / delta)

            # Evaluate the function at a new point.
            xnew = x + alpha * d
            fnew = f(xnew, *args)
            stat['f_eval'] += 1

            # Calculate the new comparison ratio.
            Delta = 2.0 * (fnew - fold) / (alpha * mu)
            if (Delta >= 0.0):
                success = True
                nsuccess += 1
                x = xnew
                fnow = fnew
                gnow = gradnew
            else:
                success = False
                fnow = fold
                gnow = gradold
            # _end_if_

            if self._diagnostics_on:
                # Total gradient.
                total_grad = np.abs(gnow).sum()

                # Store statistics.
                stat["Fx"][j] = float(fnow)
                stat["Gx"][j] = total_grad
                stat["beta"][j] = beta

                # Used in debuging mode.
                if (j % self._upd) == 0.0:
                    print(" {0}: fx={1:.3f}, sum(gx)={2:.3f}".format(j, float(fnow), total_grad))
                # _end_if_

            # _end_if_

            # TBD
            if success:
                # Check for termination.
                if (np.abs(alpha * d).max() <= self._tol_x) and (np.abs(fnew - fold) <= self._tol_fx):
                    fx = fnew
                    stat['Itr'] = j
                    return {"x_opt": x, "f_opt": fx, "stat": stat}
                else:
                    # Update variables for new position.
                    fold = fnew
                    gradold = gradnew

                    # Derivative df(x).
                    if df:
                        gradnew = df(x, *args)
                    else:
                        gradnew = numerical_derivative(f, x, method='cdf', *args)
                    # _end_if_

                    # Increase function/gradients evaluations by one.
                    stat['f_eval'] += 1
                    stat['g_eval'] += 1

                    # If the gradient is zero then we are done.
                    if (gradnew.dot(gradnew) == 0.0):
                        fx = fnow;
                        stat['Itr'] = j
                        return {"x_opt": x, "f_opt": fx, "stat": stat}
                    # _end_if_
                # _end_if_
            # _end_if_

            # Adjust beta according to comparison ratio.
            if (Delta < 0.25):
                beta = min(4.0 * beta, betaMax)
            # _end_if_

            if (Delta > 0.75):
                beta = max(0.5 * beta, betaMin)
            # _end_if_

            # Update search direction using Polak-Ribiere formula, or re-start
            # in the direction of negative gradient after 'D' steps.
            if (nsuccess == D):
                d = -gradnew
                nsuccess = 0
            else:
                if success:
                    gamma = gradnew.dot(gradold - gradnew) / mu
                    d = gamma * d - gradnew
                # _end_if_
            # _end_if_
        # _end_for_

        # Display a warning to the user.
        print(' SCG: Maximum number of iterations has been reached.')

        # Here we have reached the maximum number of iterations.
        fx = fold

        # Final return statement.
        return {"x_opt": x, "f_opt": fx, "stat": stat}
    # _end_def_

# _end_class_
