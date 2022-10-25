import pickle
import numpy as np
import random as rd

def read_cifar_batch(batchfile):
    """Returns the data and labels from a CIFAR file  using pickle module.

    Args:
        batchfile (string): The name of the batchfile.

    Returns:
        data (np.ndarray(np.float32)): The matrix data where each line represents a objects.
        labels (np.ndarray(np.int64)): The vector containing the corresponding label for each data row in data.
    """

    with open(batchfile, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data = np.array(dict[b'data'], dtype=np.float32)
    labels = np.array(dict[b'labels'], dtype=np.int64)

    return data, labels


def read_cifar(directory):
    """Concatenates and returns all data from the specified directory.

    Args:
        directory (string): The name of the directory containing the CIFAR files.

    Returns:
        data (np.ndarray(np.float32)): The matrix data where each line represents a object.
        labels (np.ndarray(np.int64)): The vector containing the corresponding label for each data row in data.
    """

    # Initialize data and labels with test data
    path_test = directory + "/test_batch"
    data, labels = read_cifar_batch(path_test)

    # Loop over train batches to concatenate data and labels
    for i in range(1, 6):
        path = directory + "/data_batch_" + str(i)
        new_data, new_labels = read_cifar_batch(path)
        data = np.concatenate((data, new_data), 0)
        labels = np.concatenate((labels, new_labels), 0)
    
    return data, labels


def split_dataset(data, labels, split_factor):
    """Randomly splits the entire dataset into a train and test dataset according to a factor.

    Args:
        data (np.ndarray(np.float32)): The matrix data where each line represents an object.
        labels (np.ndarray(np.int64)): The vector containing the corresponding label for each data row in data.
        split_factor (np.float32): The size factor between data_train and data.

    Returns:
        data_train (np.ndarray(np.float32)): The train dataset.
        labels_train (np.ndarray(np.int64)): The train labels for each train data.
        data_test (np.ndarray(np.float32)): The test dataset.
        labels_test (np.ndarray(np.int64)): The test labels for each test data.
    """

    size = np.shape(data)[0]
    test_size = int((1 - split_factor) * size)
    test_indices = rd.sample(range(size), test_size)

    # As the testing dataset is usually smaller, it is extracted from the entire dataset and not the other way aorund
    data_test = data[test_indices]
    labels_test = labels[test_indices]

    # The training dataset consist in the remaining images and labels
    data_train = np.delete(data, test_indices, 0)
    labels_train = np.delete(labels, test_indices, 0)

    return data_train, labels_train, data_test, labels_test


if __name__ == "__main__":

    # Test read_cifar_batch function
    file = "data/data_batch_1"
    data1, labels1 = read_cifar_batch(file)
    print(data1[:3], labels1[:3])

    # Test read_cifar function
    data, labels = read_cifar("data")
    print(np.shape(data), np.shape(labels))
    print(data[:3], labels[:3])

    # Check shapes of training and testing split dataset
    a, b, c, d = split_dataset(data, labels, 0.75)
    print(np.shape(a), np.shape(b), np.shape(c), np.shape(d))



