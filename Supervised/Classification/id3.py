# Specific imports.
from ml_util import split_dataset_at_feature

# Public interface.
__all__ = ['ID3']

# Current version.
__version__ = '0.0.1'

# Author.
__author__ = "Michalis Vrettas, PhD - Email: michail.vrettas@gmail.com"


# ID3 class definition.
class ID3(object):
    """
    Description:
    TBD
    """

    def __init__(self, data):
        """
        Description:
        Constructor for an ID3 object.

        Args:
        - data: list of lists.

        Raises:
        - ValueError: If the range limits (low, high) are not ordered correctly.
        """

        self._data = data
    # _end_def_

    @staticmethod
    def calc_shannon_entropy(data):
        """
        Description:
        It computes the Shannon entropy of a data set. The more organized a data set is,
        the lower the entropy value will be. Here we choose the base log2() function but
        this is not very important at the moment.

        Args:
            - data: (list of lists) input data-set.

        Note:
            - We assume that the last column in the data contains the class label.
        """
        # Sanity check.
        if not data:
            raise ValueError(" Input data set is empty.")
        # _end_if_

        # Label counter.
        label_counts = {}

        # Check all the entries in the data-set.
        for record in data:
            # Get the label of the input vector.
            this_label = record[-1]

            # If it is not in the dictionary add it.
            if this_label not in label_counts:
                label_counts[this_label] = 0
            # _end_if_

            # Increase counter.
            label_counts[this_label] += 1
        # _end_for_

        # Define the entropy variable.
        total_entropy = 0.0

        # Get the total number of data vectors.
        num_n = float(len(data))

        # Compute the entropy.
        for key in label_counts:
            prob = float(label_counts[key])/num_n
            total_entropy -= prob * np.log2(prob)
        # _end_for_

        return total_entropy
    # _end_def_

    def choose_best_feature(self, data):
        """
        Description:
        Selects the best feature to split the data set, using the entropy as a measure of goodness.

        Args:
            - data: (list of lists) input data-set.

        Note:
            - We assume that the last column in the data contains the class label.
        """

        # Number of samples in the data set.
        tot_n = len(data)

        # Initial entropy of the data set.
        entropy = self.calc_shannon_entropy(data)

        # Information gain.
        best_info_gain = 0.0

        # Best feature.
        best_feature = -1

        # Go through all the features.
        for i in range(len(data[0]) - 1):
            # Split the data set on the feature 'i'.
            sub_set = split_dataset_at_feature(data, i)

            # Entropy for the current split.
            split_entropy = 0.0

            # Calculate the combined entropy of the split.
            for j in sub_set:
                split_entropy += (len(sub_set[j])/tot_n)*self.calc_shannon_entropy(sub_set[j])
            # _end_for_

            # Compute the information gain (w.r.t. the initial entropy).
            split_info_gain = entropy - split_entropy

            # If the split has reduced the entropy update the values.
            if split_info_gain > best_info_gain:
                best_info_gain = split_info_gain
                best_feature = i
            # _end_if_

        # _end_for_

        return best_feature
    # _end_def_

# _end_class_
