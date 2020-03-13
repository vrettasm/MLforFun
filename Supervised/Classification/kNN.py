# Specific imports.
from scipy.spatial import distance

# Public interface.
__all__ = ['KNN']

# Current version.
__version__ = '0.0.1'

# Author.
__author__ = "Michalis Vrettas, PhD - Email: michail.vrettas@gmail.com"


# K-Nearest Neighbor class.
class KNN(object):
    """
    Description:
    This class implements a k-Nearest Neighbor (kNN) classification algorithm.
    The class constructor works as follows:

    1) Initialize the kNN object using a data set, along with its corresponding labels
    and optionally give a range to normalize the input data.

    2) If a range has been given use it to normalize the data.
    """

    def __init__(self, data, labels, norm_range=None):
        """
        Description:
        Constructor for a kNN object.

        Args:
        - data: numpy array with dimensions (MxN).
        - labels: list with the class of each input vector.
        - norm_range: normalization range (feature scaling) is used to bring all
        values into the range [a, b].

        Note:
        There is no default normalization. It is the user's responsibility to set
        it correctly.

        Raises:
        - ValueError: If the range limits (low, high) are not ordered correctly.
        """

        # If a range has been given process it.
        if norm_range:
            # Get the [low, high] limits of the range.
            low_lim, high_lim = norm_range

            # Sanity check.
            if low_lim >= high_lim:
                raise ValueError(" KNN: Input limits are invalid: norm_range = (low, high).")
            # _end_check_

            # Get the min/max values (of the data columns).
            d_min = data.min(0)
            d_max = data.max(0)

            # Rescale the input.
            data = low_lim + ((data - d_min) * (high_lim - low_lim)) / (d_max - d_min)
        # _end_if_

        # Check the number of input data instances against the number of labels.
        if data.shape[0] != len(labels):
            raise ValueError(" KNN: Dimensions mismatch."
                             " The number of labels is not the same as the number of input data.")
        # _end_if_

        # Store the data.
        self._data = data

        # Store the labels (classes).
        self._labels = labels

    # _end_def_

    def classify(self, x, k=1):
        """
        Description:
        Classify the input vector(s) 'x' with 'k' neighbors. The classification works as follows:
            1) For each test vector compute its Euclidean distance from ALL the vectors in the input data.
            2) Sort the distances and select the 'k' with the lowest values.
            3) Check the class label of all 'k' selected input vectors and use the majority vote to assign
            to the test vector.

        Args:
        - x: test data-set (to be classified).
        - k: number of neighbours to consider.

        Raises:
        - TypeError: If the type of neighbours is integer.
        - ValueError: If the number of neighbours is invalid.
        """
        # Check if 'k' is not integer.
        if not isinstance(k, int):
            raise TypeError(" KNN.classify: The number of neighbours 'k' should be integer.")
        # _end_if_

        # Check for negative values.
        if k < 1:
            raise ValueError(" KNN.classify: Invalid number of neighbors. k > 0.")
        # _end_if_

        # Check for out of limits.
        if k > len(self._labels):
            raise ValueError(" kNN.classify: The requested 'k' exceeds the number of input data.")
        # _end_if_

        # Compute the Euclidean distance of each vector in 'x' against all the input data.
        dist_x = distance.cdist(self._data, x, metric='euclidean')

        # Sort the distances in ascending order (and return the indexes).
        min_dist = dist_x.argsort(axis=0)

        # Return list with the classes.
        class_list = []

        # Perform the classification for each of the input vectors in 'x'.
        for i in range(min_dist.shape[1]):
            # Auxiliary dictionary.
            class_count = {}

            # Find the 'k' nearest class labels.
            for j in range(k):
                # Get the label.
                this_label = self._labels[min_dist[:, i][j]]

                # Add the label to the dictionary.
                if this_label not in class_count:
                    class_count[this_label] = 0
                # _end_if_

                # Increase instance counter by one.
                class_count[this_label] += 1
            # _end_for_

            # Sort the dictionary (reverse order) according to the number of occurrences.
            sorted_class = sorted(class_count, key=class_count.__getitem__, reverse=True)

            # Now find the majority winner class.
            class_list.append(sorted_class[0])
        # _end_for_

        # Return the list with the classification result.
        return class_list
    # _end_def_

# _end_class_
