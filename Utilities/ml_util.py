# Generic import
import numpy as np


def central_difference(f, x, *args):
    """
    Description:
    It returns the numerical derivative of input function "f"
    at the point "x", i.e. df(x)/dx, by using the central
    difference formula.

    Args:
        - f: function handle.
        - x: point to evaluate the derivative.
        - *args: additional input parameters of the function.

    Output:
        - df(x)/dx
    """
    # The optimal step when using the CDF should scale
    # with respect to the cubic root of "eps".
    cbrt_eps = (np.finfo(float).eps) ** (1.0 / 3.0)

    # Number of input parameters.
    D = x.size

    # Preallocate for efficiency.
    df = np.zeros(D)

    # Auxilliary vector.
    e = np.zeros(D)

    # Check all 'D' directions (coordinates of x).
    for i in range(D):
        # Switch ON i-th direction.
        e[i] = 1.0

        # Step size (for the i-th variable).
        h = cbrt_eps if (x[i] == 0.0) else x[i] * cbrt_eps

        # Move a small way in the i-th direction of x+h.
        fplus = f(x + h * e, *args)

        # Move a small way in the i-th direction of x-h.
        fminus = f(x - h * e, *args)

        # Central difference formula for approximation.
        df[i] = (fplus - fminus) / (2.0 * h)

        # Switch OFF i-th direction.
        e[i] = 0.0
    # _end_if_

    return df


# _end_def_

def forward_difference(f, x, *args):
    """
    Description:
    It returns the numerical derivative of input function "f"
    at the point "x", i.e. df(x)/dx, by using the forward
    difference formula.

    Args:
        - f: function handle.
        - x: point to evaluate the derivative.
        - *args: additional input parameters of the function.

    Output:
        - df(x)/dx
    """
    # Step size.
    h = 1.0E-6

    # Number of input parameters.
    D = x.size

    # Preallocate for efficiency.
    df = np.zeros(D)

    # Auxilliary vector.
    e = np.zeros(D)

    # Compute the f(x) only once.
    fx = f(x, *args)

    # Check all 'D' directions (coordinates of x).
    for i in range(D):
        # Switch ON i-th direction.
        e[i] = 1.0

        # Move a small way in the i-th direction of x+h.
        fplus = f(x + h * e, *args)

        # Central difference formula for approximation.
        df[i] = (fplus - fx) / h

        # Switch OFF i-th direction.
        e[i] = 0.0
    # _end_if_

    return df


# _end_def_

def backward_difference(f, x, *args):
    """
    Description:
    It returns the numerical derivative of input function "f"
    at the point "x", i.e. df(x)/dx, by using the backward
    difference formula.

    Args:
        - f: function handle.
        - x: point to evaluate the derivative.
        - *args: additional input parameters of the function.

    Output:
        - df(x)/dx
    """
    # Step size.
    h = 1.0E-6

    # Number of input parameters.
    D = x.size

    # Preallocate for efficiency.
    df = np.zeros(D)

    # Auxilliary vector.
    e = np.zeros(D)

    # Compute the f(x) only once.
    fx = f(x, *args)

    # Check all 'D' directions (coordinates of x).
    for i in range(D):
        # Switch ON i-th direction.
        e[i] = 1.0

        # Move a small way in the i-th direction of x+h.
        fminus = f(x - h * e, *args)

        # Central difference formula for approximation.
        df[i] = (fx - fminus) / h

        # Switch OFF i-th direction.
        e[i] = 0.0
    # _end_if_

    return df


# _end_def_

def numerical_derivative(f, x, method='cdf', *args):
    """
    Description:
    It returns the numerical derivative of input function "f"
    at the point "x", i.e. df(x)/dx, by calling the relevant
    numerical method to compute the derivative.

    Args:
        - f: function handle.
        - x: point to evaluate the derivative.
        - method: numerical integration (default = 'cdf')
        - args: additional input parameters of the function.

    Raises:
        - ValueError: If the input method is not recognized.
    """

    # Call the right method or throw an error.
    if method == "cdf":
        return central_difference(f, x, *args)
    elif method == "fwd":
        return forward_difference(f, x, *args)
    elif method == "bwd":
        return backward_difference(f, x, *args)
    else:
        raise ValueError(" Unknown method of differentiation.")


# _end_def_

def flat_list(x=None):
    """
    Description:
    It returns a list that contains all the elements form the input list of lists.
    It should work for any number of levels.

    Example:
        >>> x = flat_list([1, 'k', [], 3, [4, 5, 6], [[7, 8]], [[[9]]]])
        >>> x
        >>> [1, 'k', 3, 4, 5, 6, 7, 8, 9]

    Args:
        - x (list): List of lists of objects.

    Raises:
        - TypeError: If the input is not a list object.
    """

    # Check for empty input.
    if x is None:
        return []
    # _end_if_

    # Input 'x' should be a list.
    if not isinstance(x, list):
        raise TypeError(" Input should be a list.")
    # _end_if_

    # Define the return list.
    flat_x = []

    # Go through all the list elements.
    for item_x in x:
        # Check for embedded lists.
        if isinstance(item_x, list):
            # Check for empty entries.
            if not item_x:
                continue
            # _end_if_

            # Note the recursive call in "flat_list".
            for ix in flat_list(item_x):
                flat_x.append(ix)
            # _end_for_
        else:
            # If the item is not a list.
            flat_x.append(item_x)
        # _end_if_
    # _end_for_

    # Return the flatten list.
    return flat_x


# _end_def_

def split_dataset_at_feature(data, axis=0):
    """
    Description:
    Splits the input data set at the requested feature, given by the index number 'idx'.

    Args:
        - data: (list of lists) input data-set.
        - axis: feature index to split the data-set.

    Note:
        Default index is '0' (i.e. the first feature of the data set).

    Raises:
        ValueError if the index is out of bounds.
    """

    # Data must be a list.
    if not isinstance(data, list):
        raise TypeError(" Input data should be a list.")
    # _end_if_

    # Index must be integer.
    if not isinstance(axis, int):
        raise TypeError(" Axis value should be integer.")
    # _end_if_

    # Get the number of features.
    # Note:  The last column is assumed to contain the class label,
    # therefore it should never be included for splitting the data.
    n_features = len(data[0]) - 1

    # If there is only one feature, there is nothing to split.
    if n_features == 1:
        return data
    # _end_if_

    # Sanity check.
    if axis < 0 or axis >= n_features:
        raise ValueError(" Feature index is out of bounds.")
    # _end_if_

    # Every entry in the dictionary will contain a sub-set of the
    # original data-set, split at a value of the feature.
    partition_data = {}

    # Split the data-set into sub-sets.
    for record in data:
        # First part of the record.
        part_1 = record[:axis]

        # Second part of the record.
        part_2 = record[axis + 1:]

        # Combine the two parts in one record.
        part_1.extend(part_2)

        # Construct a key (string) entry.
        key = "chID_{0}:{1}".format(idx, record[axis])

        # Check if this key has been seen.
        if key not in node_dict.keys():
            # If not then add it to the dictionary.
            partition_data[key] = []
        # _end_if_

        # Add the record on the dictionary.
        partition_data[key].append(part_1)
    # _end_for_

    return partition_data

# _end_def_
