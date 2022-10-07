from os import getcwd, getcwdb
import pickle
from matplotlib import test
import numpy as np
import random 


def read_cifar_batch(batchfile):
    # Returns the data and labels corresponding to a single data batch
    with open(batchfile, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data = np.array(dict[b'data'], dtype=np.float32)
    labels = np.array(dict[b'labels'], dtype=np.int64)

    return data, labels


def read_cifar(directory):
    path_test = directory + "/test_batch"
    data, labels = read_cifar_batch(path_test)
    for i in range(1, 6):
        path = directory + "/data_batch_" + str(i)
        new_data, new_labels = read_cifar_batch(path)
        data = np.concatenate((data, new_data), 0)
        labels = np.concatenate((labels, new_labels), 0)

    return data, labels


def split_dataset(data, labels, split):
    size = np.shape(data)[0]
    test_size = int((1 - split) * size)

    test_indices = random.sample(range(size), test_size)
    #print(test_indices, len(test_indices))

    data_test = data[test_indices]
    labels_test = labels[test_indices]

    data_train = np.delete(data, test_indices, 0)
    labels_train = np.delete(labels, test_indices, 0)

    return data_train, labels_train, data_test, labels_test







if __name__ == "__main__":

    file = "data/data_batch_1"
    
    d1, l1 = read_cifar_batch(file)

    data, labels = read_cifar("data")
    #print(np.shape(data), np.shape(labels))

    print(np.shape(data))
    a, b, c, d = split_dataset(data, labels, 0.25)

    print(np.shape(a), np.shape(b), np.shape(c), np.shape(d))





