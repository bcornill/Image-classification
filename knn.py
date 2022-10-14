import numpy as np
import matplotlib.pyplot as plt
from read_cifar import *

# Compute the Euclidian distance between 2 matrices
def distance_matrix(M1, M2):
    dists = np.square(M1) - 2 * M1 * M2 + np.square(M2)
    dist = np.sqrt(np.sum(dists))

    return dist


# Compute the matrix of distances between all elements of one dataset and all the elements of another
def sets_distance(s1, s2):
    dists = np.zeros((len(s1), len(s2)))
    
    for i in range(len(s1)):
        for j in range(len(s2)):
            dists[i, j] = distance_matrix(s1[i], s2[j])

    return dists


# Return the k-nearest neighbors and the predicted label for each row entry of the dists matrix
def knn_predict(dists, labels_train, k):
    test_size, _ = np.shape(dists)

    # Get the indixes and corresponding labels of the k-nearest neighbors
    indices_array = np.argpartition(dists, k)
    k_nearest_indices = indices_array[:,:k]
    knn = [[labels_train[index] for index in k_nearest_indices[i]] for i in range(test_size)]

    # The predicted label is the one with the most occurences in the k-nearest neighbors
    predictions = [max(set(l), key = l.count) for l in knn]

    return knn, predictions


# Compute the accuracy of the KNN algorithm by comparing the predicted labels to the actual ones
def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    dists = sets_distance(data_test, data_train)
    _, predictions = knn_predict(dists, labels_train, k)
    
    accuracy = np.count_nonzero((predictions - labels_test) == 0) * 100 / len(labels_test)

    return round(accuracy, 2)


if __name__ == "__main__":

    # Test evaluate_knn on smaller datasets to establish algoithm correctness and closure
    data, labels = read_cifar("data")
    split_factor = 0.75
    a, b, c, d = split_dataset(data, labels, split_factor)
    a, b, c, d = a[:4500], b[:4500], c[:1500], d[:1500]
    k = 5

    prec = evaluate_knn(a, b, c, d, k)
    print("Précision de l'algorithme knn : " + str(round(prec, 1)) + "%")

    # Plot the accuracy of the KNN algorithm on the full datasets while changing the number of neighbors considered
    data, labels = read_cifar("data")
    split_factor = 0.9
    a, b, c, d = split_dataset(data, labels, split_factor)
    k = [i for i in range(1, 21)]
    acc = [evaluate_knn(a, b, c, d, i) for i in k]
    
    plt.plot(k, acc)
    plt.xlabel("Nombre de plus proches voisins")
    plt.ylabel("Précision (en %)")
    plt.title("Efficacité de l'algorithme KNN en fonction du nombre de voisins considérés")
    plt.show()






