# Generic imports.
import numpy as np

# Public interface.
__all__ = ['PatternSearch']

# Current version.
__version__ = '0.0.1'

# Author.
__author__ = "Michalis Vrettas, PhD - Email: michail.vrettas@gmail.com"


# Pattern Search optimization class definition.
class PatternSearch(object):
    """
    Description:
    This class implements a Pattern Search (derivative free) optimization object.

    References:
    (1) Hooke, R.; Jeeves, T.A. (1961). "'Direct search' solution of numerical and statistical problems".
        Journal of the ACM. 8 (2): 212–229. doi:10.1145/321062.321069.

    (2) Dolan, E.D.; Lewis, R.M.; Torczon, V.J. (2003). "On the local convergence of pattern search".
        SIAM Journal on Optimization. 14 (2): 567–583. CiteSeerX 10.1.1.78.2407. doi:10.1137/S1052623400374495.

    (3) Powell, Michael J. D. 1973. ”On Search Directions for Minimization Algorithms.”
        Mathematical Programming 4: 193—201.
    """

    def __init__(self, maxit=1500, tol_x=1.0E-6, update=10, diagnostics_on=False):
        """
        Description:
        Constructor for an PatternSearch object.

        Args:
        - maxit (int): maximum number of iterations.
        - tol_x (float): precision in 'x' (input) space.
        - update (int): frequency to update the output and store the statistics.
        - diagnostics_on (bool): flag that signals the monitoring of the algorithm stats.

        Raises:
        - ValueError: if some input parameter is out of range.
        - TypeError: if some input parameter has the wrong type.
        """

        # Check the input for correct type.
        if isinstance(maxit, int):
            # Check the input for correct range.
            if maxit > 0:
                # Assing the value.
                self._maxit = maxit
            else:
                raise ValueError(" PatternSearch: Number of max iterations must be > 0.")
        else:
            raise TypeError(" PatternSearch: Number of max iterations must be integer.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(tol_x, float):
            # Check the input for correct range.
            if tol_x > 0.0:
                # Assing the value.
                self._tol_x = tol_x
            else:
                raise ValueError(" PatternSearch: Precision in 'x' must be > 0.")
        else:
            raise TypeError(" PatternSearch: Precision in 'x' must be float.")
        # _end_if_

        # Check the input for correct type.
        if isinstance(update, int):
            # Check the input for correct range.
            if update > 0:
                # Assing the value.
                self._upd = update
            else:
                raise ValueError(" PatternSearch: Number of update must be > 0.")
        else:
            raise TypeError(" PatternSearch: Number of update must be integer.")
        # _end_if_

        # Check the input for correct type.
        if type(diagnostics_on) is bool:
            # Assing the value.
            self._diagnostics_on = diagnostics_on
        else:
            raise TypeError(" PatternSearch: Diagnostics flag must be bool.")
        # _end_if_

    # _end_def_

    @property
    def maxit(self):
        """
        Description:
        Accessor (get) for the maximum number of iterations.
        """
        return self._maxit

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

    @maxit.setter
    def maxit(self, n):
        """
        Description:
        Accessor (set) for the maximum number of iterations.
        """
        # Type check.
        if isinstance(n, int):
            # Range check.
            if n > 0:
                self._maxit = n
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

    def optimize(self, f, x0, radius=1.0, limits=(-np.inf, np.inf), *args):
        """
        Description:
            Pattern Search optimization, attempts to find a local minimum of the function f(x).
            Here 'x0' is a column vector and 'f' returns a scalar value. The minimisation process
            works as follows:

            At each iteration, the pattern either moves to the point which best minimizes its
            objective function, or shrinks in size if no point is better than the current point,
            until the desired accuracy has been achieved, or the algorithm reaches a predetermined
            number of iterations.

        Input parameters:
            f      (callable): is the objective function to be optimised.
            x0     (np.array): is the initial point of function (D x 1).
            radius (float): the initial radius of each parameter. It can either be scalar value
                          : or an array of the same size as x0.
            limits (tuple): constraints of the parameter space in a tuple (Upper, Lower).
                          : Upper, can be a number or a np.array with the same size as x0.
                          : Lower, can be a number or a np.array with the same size as x0.
            *args         : additional parameters for both 'f'.

        Output parameters:
            xmin   : the point where the minimum was found.
            fmin   : the function value, at the minimum point.
            stat   : statistics that collected through the optimisation process.
        """

        # Display method name.
        print(" >> PatternSearch optimization ")

        # Ensure compatibility.
        x0 = np.asfarray(x0).flatten()

        # Dimensionality of the input vector.
        L = x0.size

        # Statistics dictionary.
        stat = {'Itr': self._maxit, 'f_eval': 0, 'shrink': 0}

        # If the diagnostics is on we add more fields.
        if self._diagnostics_on:
            stat['Fx'] = np.zeros(self._maxit)
        # _end_if_

        # Initialization.
        xmin = np.asfarray(x0).copy()

        # First function evaluation.
        fmin = f(xmin, *args)

        # Increase function evaluations.
        stat['f_eval'] += 1

        # Auxilliary vector.
        e = np.zeros(L)

        # Main optimization loop.
        for i in range(self._maxit):
            # Make a copy of the current minimum.
            x = xmin.copy()

            # Initialize boolean flag.
            found_min = False

            # Evaluate the function at the grid points
            # within the limits for each parameter.
            for k in range(L):
                # Switch ON k-th direction.
                e[k] = 1.0

                # Temporary position.
                xk = x + (e * radius)

                # Check (upper) bounds.
                if np.all(xk <= limits[1]):
                    # Evaluate the function in the '+' direction.
                    f_more = f(xk, *args)

                    # Increase function evaluations.
                    stat['f_eval'] += 1

                    # Check for minimum.
                    if np.isfinite(f_more) and (f_more < fmin):
                        # Update the flag.
                        found_min = True

                        # Update for minimum.
                        fmin = f_more
                        xmin = xk.copy()
                    # _end_if_
                # _end_if_

                # Temporary position.
                xk = x - (e * radius)

                # Check (lower) bounds.
                if np.all(xk >= limits[0]):
                    # Evaluate the function in the '-' direction.
                    f_less = f(xk, *args)

                    # Increase function evaluations.
                    stat['f_eval'] += 1

                    # Check for minimum.
                    if np.isfinite(f_less) and (f_less < fmin):
                        # Update the flag.
                        found_min = True

                        # Update for minimum.
                        fmin = f_less
                        xmin = xk.copy()
                    # _end_if_
                # _end_if_

                # Switch OFF k-th direction.
                e[k] = 0.0
            # _end_for_

            # Check for diagnostics.
            if self._diagnostics_on:
                # Store the current fmin.
                stat['Fx'][i] = fmin

                # Display current info:
                if (i % self._upd) == 0.0:
                    print(" {0}: fmin={1:.3f}, shrinked={2} times".
                          format(i, float(fmin), stat['shrink']))
                # _end_if_
            # _end_if_

            # Check for convergence.
            if np.all(2.0 * radius < self._tol_x):
                stat['Itr'] = i
                return {"x_opt": xmin, "f_opt": fmin, "stat": stat}
            # _end_if_

            # Shrink by predefined value.
            if not found_min:
                radius *= 0.5
                stat['shrink'] += 1
            # _end_if_

        # _end_main_loop_

        # Display a warning to the user.
        print(' PatternSearch: Maximum number of iterations has been reached.')

        # Final return statement.
        return {"x_opt": xmin, "f_opt": fmin, "stat": stat}
    # _end_def_

# _end_class_
