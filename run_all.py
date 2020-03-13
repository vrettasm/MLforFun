# General import.
import numpy as np
from sklearn import datasets


def test_knn_classification():
    # Local import.
    from Supervised.Classification.kNN import KNN

    # Get the Iris data-set.
    iris = datasets.load_iris()

    # Convert the labels to list.
    labels = iris.target.tolist()

    # Test samples.
    x = np.reshape(iris.data[48:53], (5, 4))

    # Create the kNN object.
    knn = KNN(iris.data, labels)

    # Classification results.
    x_label = knn.classify(x, k=5)

    # Print the results.
    print(" Classification result: ", x_label)

    # Print the true classes.
    print(" True classes         : ", labels[48:53])

# _end_classification_test_


# Main function.
if __name__ == '__main__':
    test_knn_classification()
# _end_if_
