# Generic imports.
import numpy as np
from ml_util import numerical_derivative

# Public interface.
__all__ = ['ConjugateGradient']

# Current version.
__version__ = '0.0.1'

# Author.
__author__ = "Michalis Vrettas, PhD - Email: michail.vrettas@gmail.com"

# Conjugate Gradient optimization class definition.
class ConjugateGradient(object):
    """
    Description:
    This class implements a Conjugate Gradient optimization object.

    NOTE: This code is adopted from NETLAB (a free MATLAB library).

    Reference Book:
    (1) Ian T. Nabney (2001): Netlab: Algorithms for Pattern Recognition.
        Advances in Pattern Recognition, Springer. ISBN: 1-85233-440-1.
    """

    def __init__(self, nit=500, tol_x=1.0E-6, tol_fx=1.0E-8, update=100, diagnostics_on=False):
        """
        Description:
        Constructor for an CG object.

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
                raise ValueError(" CG: Number of max iterations must be > 0.")
        else:
            raise TypeError(" CG: Number of max iterations must be integer.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(tol_x, float):
            # Check the input for correct range.
            if tol_x > 0.0:
                # Assing the value.
                self._tol_x = tol_x
            else:
                raise ValueError(" CG: Precision in 'x' must be > 0.")
        else:
            raise TypeError(" CG: Precision in 'x' must be float.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(tol_fx, float):
            # Check the input for correct range.
            if tol_fx > 0.0:
                # Assing the value.
                self._tol_fx = tol_fx
            else:
                raise ValueError(" CG: Precision in 'fx' must be > 0.")
        else:
            raise TypeError(" CG: Precision in 'fx' must be float.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(update, int):
            # Check the input for correct range.
            if update > 0:
                # Assing the value.
                self._upd = update
            else:
                raise ValueError(" CG: Number of update must be > 0.")
        else:
            raise TypeError(" CG: Number of update must be integer.")
        # _end_if_

        # Check the input for correct type.
        if type(diagnostics_on) is bool:
            # Assing the value.
            self._diagnostics_on = diagnostics_on
        else:
            raise TypeError(" CG: Diagnostics flag must be bool.")
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
                raise ValueError(" CG: Number of max iterations must be > 0.")
            # _end_if_
        else:
            raise TypeError(" CG: Type of max iterations must be int.")
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
                raise ValueError(" CG: Precision for convergence in 'x' must be > 0.")
            # _end_if_
        else:
            raise TypeError(" CG: Precision in 'x' must be of type float.")
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
                raise ValueError(" CG: Precision for convergence in 'f(x)' must be > 0.")
            # _end_if_
        else:
            raise TypeError(" CG: Precision in 'f(x)' must be of type float.")
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
                raise ValueError(" CG: Update frequency must be > 0.")
            # _end_if_
        else:
            raise TypeError(" CG: Type of update frequency must be int.")
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
            raise TypeError(" CG: Type of diagnostics flag is bool.")
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
        print(" >> CG optimization (with backtracking)")

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
            stat['eta'] = np.zeros(self._nit)
        # _end_if_

        # Initialization.
        x = np.asfarray(x0).copy()

        # Initial learning rate: must be positive.
        eta0 = 0.15

        # Decrease rate for the step size.
        r = 0.5

        # Initial function/gradients value.
        fnew = f(x, *args)

        # If a derivative function has been given use it.
        if df:
            gradnew = df(x, *args)
        else:
            gradnew = numerical_derivative(f, x, method='cdf', *args)
        # _end_if_

        # Increase function/gradient evaluations by one.
        stat['f_eval'] += 1
        stat['g_eval'] += 1

        # Setup the initial search direction.
        d = -gradnew

        # If the diagnostics is on we add more fields.
        if self._diagnostics_on:
            stat['Fx'][0] = float(fnew)
            stat['Gx'][0] = eta0
            stat['eta'][0] = np.abs(gradnew).sum()
        # _end_if_

        # Get the machine precision constant.
        eps_float = np.finfo(float).eps

        # Main optimization loop.
        for j in range(1, self._nit):
            # Keep the old values.
            xold = x
            fold = fnew
            gradold = gradnew

            # Check if the gradient is zero.
            mu = gradold.dot(gradold)
            if (mu == 0.0):
                fx = fold
                stat['Itr'] = j
                return {"x_opt": x, "f_opt": fx, "stat": stat}
            # _end_if_

            # This shouldn't occur, but rest of code depends on 'd' being downhill.
            if (gradnew.dot(d) > 0.0):
                d = -d
            # _end_if_

            # Update search direction.
            line_sd = d/np.linalg.norm(d)

            # Try to find optimum stepsize.
            eta, cnt = self.backtrack(f, xold, fold, line_sd, eta0, r, *args)

            # Update the function evaluations.
            stat['f_eval'] += cnt

            # Exit if you can't find any better eta.
            if (eta == 0.0):
                x = xold
                fx = fold
                stat['Itr'] = j
                return {"x_opt": x, "f_opt": fx, "stat": stat}
            # _end_if_

            # Set x and fnew to be the actual search point we have found.
            x = xold + eta*line_sd

            # Evaluate function at the new point.
            fnew = f(x, *args)

            # Evaluate derivative at the new point.
            if df:
                gradnew = df(x, *args)
            else:
                gradnew = numerical_derivative(f, x, method='cdf', *args)
            # _end_if_

            # Increase function/gradient evaluations by one.
            stat['f_eval'] += 1
            stat['g_eval'] += 1

            # Check for termination.
            if (np.abs(x-xold).max() <= self._tol_x) and (np.abs(fnew-fold) <= self._tol_fx):
                fx = fnew
                stat['Itr'] = j
                return {"x_opt": x, "f_opt": fx, "stat": stat}
            # _end_if_

            # Use Polak-Ribiere formula to update search direction.
            gamma = gradnew.dot(gradold - gradnew)/mu
            d = gamma*d - gradnew

            # Used in debuging mode.
            if self._diagnostics_on:
                # Total gradient.
                total_grad = np.abs(gradnew).sum()

                # Store statistics
                stat['Fx'][j] = float(fnew)
                stat['Gx'][j] = total_grad
                stat['eta'][j] = eta

                # Used in debuging mode.
                if (j%self._upd == 0.0):
                    print(' {0}: fx={1:.3f}, sum(gx)={2:.3f}'.format(j, float(fnew), total_grad))
                # _end_if_
            # _end_if_

        # _end_for_

        # Display a warning to the user.
        print(' CG: Maximum number of iterations has been reached.')

        # Here we have reached the maximum number of iterations.
        fx = fold

        # Final return statement.
        return {"x_opt": x, "f_opt": fx, "stat": stat}
    # _end_def_

    @staticmethod
    def backtrack(f, x0, f0, df0, eta0, r, *args):
        """
            BACTRACK

        Description:
        Backtracking method to find optimum step size
        for the conjugate gradient algorithm (OPTIM_CG).

        Input parameters:
            -f     : is the objective function to be optimised.
            -x0    : is the current search point of function (D x 1).
            -fx0   : is the value of the objective function 'f',
                     at x0, i.e. f0 = f(x0) (1 x 1).
            -df0   : is the value of the gradient function 'df',
                     at x0, i.e. df0 = df(x0) (D x 1).
            -eta0  : current stepsize 0 < eta < 1.
            -r     : decrease ratio for step size 0 < r < 1.
            -*args : additional parameters for function 'f'.

        Output parameters:
            -eta   : optimal step size.
            -cnt   : number of function evaluations.

        See also: SCG
        """

        # Maximum number of trials.
        maxiter = 15

        # Optimum step size.
        eta = eta0

        # Counter.
        cnt = 0

        # Evaluate the function.
        fx = f((x0 + eta0*df0), *args)

        # Termination condition for backtracking.
        while ((cnt < maxiter) and (not np.isfinite(fx) or (fx > f0))):
            # Decrease stepsize.
            eta *= r

            # Compute the new position.
            x = x0 + eta*df0

            # Evaluate the function.
            fx = f(x, *args)

            # Increase counter by one.
            cnt += 1
        # _end_while_

        # Safeguard:
        return (max(eta, 0.0), cnt+1)
    # _end_def_

# _end_class_
