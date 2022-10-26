import numpy as np
import matplotlib.pyplot as plt
from read_cifar import *

def distance_matrix(M1, M2):
    """Returns the L2 Euclidean distance matrix between two matrices of shape (N1, N2) and (N2, N3).

    Args:
        M1 (np.ndarray(np.float32)): The matrix of shape (N1, N2).
        M2 (np.ndarray(np.float32)): The matrix of shape (N2, N3).

    Returns:
        np.ndarray(np.float32): The L2 Euclidean distance matrix between M1 and M2, of shape (N1, N3).
    """

    M1_sq_sum = np.sum(M1*M1, axis=1, keepdims=True)
    M2_sq_sum = np.sum(M2*M2, axis=1, keepdims=True)
    product = np.matmul(M1, M2.T)

    dists = np.sqrt(M1_sq_sum + M2_sq_sum.T - 2 * product)
    return dists


def knn_predict(dists, labels_train, k):
    """Returns the predicted label for each row entry of the dists matrix based on the majority of labels amongst the k-nearest neighbors.

    Args:
        dists (np.ndarray(np.float32)): The matrix of distances between the test dataset and the train dataset of shape (N1, N2).
        labels_train (np.ndarray(np.int64)): The vector of labels of the train dataset, of length N2.
        k (np.int64): The number of nearest neighbors to consider when predicting the label.

    Returns:
        np.ndarray(np.int64): The vector of predicted labels for the test dataset, of length N1.
    """

    indexes_knn = np.argsort(dists, axis=1)[:, :k]
    labels_knn = labels_train[indexes_knn]
    predicted_labels = [np.bincount(labels_knn[i]).argmax() for i in range(len(labels_knn))]

    return predicted_labels


def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    """Computes the accuracy of the KNN algorithm by comparing the predicted labels to the actual ones.

    Args:
        data_train (np.ndarray(np.float32)): The train dataset.
        labels_train (np.ndarray(np.int64)): The train labels for each train data.
        data_test (np.ndarray(np.float32)): The test dataset.
        labels_test (np.ndarray(np.int64)): The test labels for each test data.
        k (np.int64): The number of nearest neighbors to consider when predicting the label.

    Returns:
        np.int64: The accuracy of the KNN algorithm for the passed datasets.
    """

    dists = distance_matrix(data_test, data_train)
    predictions = knn_predict(dists, labels_train, k)
    accuracy = np.count_nonzero((predictions - labels_test) == 0) * 100 / len(labels_test)

    return round(accuracy, 2)


if __name__ == "__main__":
    
    # Test evaluate_knn on smaller datasets to establish algorithm correctness and closure
    data, labels = read_cifar("data")
    split_factor = 0.75
    a, b, c, d = split_dataset(data, labels, split_factor)
    a, b, c, d = a[:4500], b[:4500], c[:1500], d[:1500]
    k = 7

    prec = evaluate_knn(a, b, c, d, k)
    print("Précision de l'algorithme knn : " + str(round(prec, 1)) + "%")
    
    # Plot the accuracy of the KNN algorithm on the full datasets while changing the number of neighbors considered
    data, labels = read_cifar("data")
    split_factor = 0.9
    a, b, c, d = split_dataset(data, labels, split_factor)
    #a, b, c, d = a[:4500], b[:4500], c[:1500], d[:1500]

    k = []
    acc = []

    for i in range(1, 21):
        print("Itération " + str(i))
        k.append(i)
        acc.append(evaluate_knn(a, b, c, d, i))
    
    plt.plot(k, acc)
    plt.xlabel("Number of k-nearest neighbors")
    plt.ylabel("Accuracy (in %)")
    plt.title("Accuracy of the KNN algorithm depending on the number of neighbors considered")
    plt.show()

