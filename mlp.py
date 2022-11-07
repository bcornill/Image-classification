import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from read_cifar import *

"""
Let us note s the sigmoid function. 
For all x in R, s(x) = 1 / (1 + exp(-x))
Then, s'(x) = exp(-x) / (1 + exp(-x))^2
            = ((1 + exp(-x)) - 1) / (1 + exp(-x))^2
            = 1 / (1 + exp(-x)) - 1 / (1 + exp(-x))^2
            = (1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x)))
            = s(x) * (1 - s(x))
Thus, s' = s * (1 - s)

Then, we express the different partial derivatives of the cost function in order to compute the gradient.

2. /frac{/partial C}{/partial A^{(2)}} = 2 / N_{out} * (A^{(2)} - Y)

3. /frac{/partial C}{/partial Z^{(2)}} = /frac{/partial C}{/partial A^{(2)}} * A^{(2)} * (1 - A^{(2)})

4. /frac{/partial C}{/partial W^{(2)}} = /frac{/partial C}{/partial Z^{(2)}} * /frac{/partial Z^{(2)}}{/partial W^{(2)}}
                                      = /frac{/partial C}{/partial Z^{(2)}} * A^{(1)}

5. /frac{/partial C}{/partial B^{(2)}} = /frac{/partial C}{/partial Z^{(2)}} * /frac{/partial Z^{(2)}}{/partial B^{(2)}}
                                       = /frac{/partial C}{/partial Z^{(2)}}

6. /frac{/partial C}{/partial A^{(1)}} = /frac{/partial C}{/partial Z^{(2)}} * /frac{/partial Z^{(2)}}{/partial A^{(1)}}
                                       = /frac{/partial C}{/partial Z^{(2)}} * W^{(2)}

7. /frac{/partial C}{/partial Z^{(1)}} = /frac{/partial C}{/partial A^{(1)}} * A^{(1)} * (1 - A^{(1)})

8. /frac{/partial C}{/partial W^{(1)}} = /frac{/partial C}{/partial Z^{(1)}} * /frac{/partial Z^{(1)}}{/partial W^{(1)}}
                                       = /frac{/partial C}{/partial Z^{(1)}} * A^{(0)}

9. /frac{/partial C}{/partial B^{(1)}} = /frac{/partial C}{/partial Z^{(1)}} * /frac{/partial Z^{(1)}}{/partial B^{(1)}}
                                       = /frac{/partial C}{/partial Z^{(1)}}
"""

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    """Performs a descent gradient step and computes the Mean Square Error loss.

    Args:
        w1 (np.ndarray(np.float32)): The weight matrix of the first layer of shape (d_in, d_h).
        b1 (np.ndarray(np.float32)): The bias vector of the first layer of length d_h.
        w2 (np.ndarray(np.float32)): The weight matrix of the output layer of shape (d_h, d_out).
        b2 (np.ndarray(np.float32)): The bias vector of the output layer of length d_out.
        data (np.ndarray(np.float32)): The input data matrix of shape (batch_size, d_in).
        targets (np.ndarray(np.float32)): The ideal output matrix of shape (batch_size, d_out).
        learning_rate (np.float32): The learning rate of the network.

    Returns:
        w1 (np.ndarray(np.float32)): The updated weight matrix of the first layer of shape (d_in, d_h).
        b1 (np.ndarray(np.float32)): The updated bias vector of the first layer of length d_h.
        w2 (np.ndarray(np.float32)): The updated weight matrix of the output layer of shape (d_h, d_out).
        b2 (np.ndarray(np.float32)): The updated bias vector of the output layer of length d_out.
        loss (np.float32): The loss of the gradient descent step.
    """

    N_out = targets.shape[1]

    # Forward pass
    a0 = data  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
    predictions = a2  # the predicted values are the outputs of the output layer

    # Computing the loss (MSE)
    loss = np.mean(np.square(predictions - targets))

    # Computing the gradient of the loss function as defined above
    dz2 = 2 / N_out * (a2 - targets) * (a2 * (1 - a2))
    dw2 = np.dot(a1.T, dz2)
    db2 = np.mean(dz2, axis=0)

    dz1 = np.dot(dz2, w2.T) * a1 * (1 - a1)
    dw1 = np.dot(a0.T, dz1)
    db1 = np.mean(dz1, axis=0)

    # Updating the weights and bias matrices
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2
    
    return w1, b1, w2, b2, loss


def one_hot(labels):
    """Returns the one-hot matrix corresponding to the input labels vector.

    Args:
        labels (np.ndarray(np.int64)): The labels vector of length N.

    Returns:
        np.ndarray(np.int64): The one-hot matrix corresponding to the labels vector, of shape (label_max, N).
    """

    # Each element of the input is replaced with a vector of length max(labels) + 1
    dim = np.max(labels) + 1
    one_hot = np.eye(dim)[labels]

    return one_hot


def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    """Performs a descent gradient step computing the loss with Binary Cross-Entropy.

    Args:
        w1 (np.ndarray(np.float32)): The weight matrix of the first layer of shape (d_in, d_h).
        b1 (np.ndarray(np.float32)): The bias vector of the first layer of length d_h.
        w2 (np.ndarray(np.float32)): The weight matrix of the output layer of shape (d_h, d_out).
        b2 (np.ndarray(np.float32)): The bias vector of the output layer of length d_out.
        data (np.ndarray(np.float32)): The input data matrix of shape (batch_size, d_in).
        labels_train (np.ndarray(np.float32)): The label vector of length batch_size.
        learning_rate (np.float32): The learning rate of the network.

    Returns:
        w1 (np.ndarray(np.float32)): The updated weight matrix of the first layer of shape (d_in, d_h).
        b1 (np.ndarray(np.float32)): The updated bias vector of the first layer of length d_h.
        w2 (np.ndarray(np.float32)): The updated weight matrix of the output layer of shape (d_h, d_out).
        b2 (np.ndarray(np.float32)): The updated bias vector of the output layer of length d_out.
        loss (np.float32): The loss of the gradient descent step.
    """

    # Forward pass
    a0 = data  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)  # output of the output layer (softmax activation function)
    predictions = a2  # the predicted values are the outputs of the output layer

    # Computing the loss
    targets = one_hot(labels_train)
    batch_size = targets.shape[0]
    loss = -np.mean(targets * np.log(predictions))

    # Computing the gradient of the loss function as defined above and admiting that
    # /frac{partial C}{partial Z^{(2)}} = A^{(2)} - Y
    dz2 = a2 - targets
    dw2 = (np.dot(a1.T, dz2) / batch_size)
    db2 = np.mean(dz2, axis=0)

    dz1 = np.dot(dz2, w2.T) * a1 * (1 - a1)
    dw1 = np.dot(a0.T, dz1) / batch_size
    db1 = np.mean(dz1, axis=0)

    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2

    return w1, b1, w2, b2, loss


def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    """Performs num_epoch training steps of the Neural Network.

    Args:
        w1 (np.ndarray(np.float32)): The weight matrix of the first layer of shape (d_in, d_h).
        b1 (np.ndarray(np.float32)): The bias vector of the first layer of length d_h.
        w2 (np.ndarray(np.float32)): The weight matrix of the output layer of shape (d_h, d_out).
        b2 (np.ndarray(np.float32)): The bias vector of the output layer of length d_out.
        data_train (np.ndarray(np.float32)): The train dataset matrix of shape (batch_size, d_in).
        labels_train (np.ndarray(np.float32)): The label vector of length batch_size.
        learning_rate (np.float32): The learning rate of the network.
        num_epoch (np.int64): The number of steps used to train the Neural Network.

    Returns:
        w1 (np.ndarray(np.float32)): The updated weight matrix of the first layer of shape (d_in, d_h).
        b1 (np.ndarray(np.float32)): The updated bias vector of the first layer of length d_h.
        w2 (np.ndarray(np.float32)): The updated weight matrix of the output layer of shape (d_h, d_out).
        b2 (np.ndarray(np.float32)): The updated bias vector of the output layer of length d_out.
        train_accuracies (np.ndarray(np.float32)): The vector of length num_epoch of the network's accuracy at each step.
    """

    train_accuracies = np.zeros((num_epoch, 1))

    for k in range(num_epoch):
        w1, b1, w2, b2, _ = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        accuracy = test_mlp(w1, b1, w2, b2, data_train, labels_train)
        train_accuracies[k] = accuracy
        print("Accuracy at step " + str(k) + ": " + str(accuracy) + "%")

    return train_accuracies


def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    """Tests the network on the test set.

    Args:
        w1 (np.ndarray(np.float32)): The weight matrix of the first layer of shape (d_in, d_h).
        b1 (np.ndarray(np.float32)): The bias vector of the first layer of length d_h.
        w2 (np.ndarray(np.float32)): The weight matrix of the output layer of shape (d_h, d_out).
        b2 (np.ndarray(np.float32)): The bias vector of the output layer of length d_out.
        data_test (np.ndarray(np.float32)): The test dataset matrix of shape (batch_size, d_in).
        labels_test (np.ndarray(np.float32)): The label vector of length batch_size for each image of the dataset.

    Returns
        np.float64: The accuracy of the network.
    """

    # Forward pass
    a0 = data_test  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    # output of the hidden layer (sigmoid activation function)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    # output of the output layer (sigmoid activation function)
    a2 = 1 / (1 + np.exp(-z2))
    predictions = a2  # the predicted values are the outputs of the output layer

    predicted_labels = np.argmax(predictions, axis=1)#, keepdims=True)
    accuracy = np.count_nonzero(predicted_labels == labels_test) * 100 / len(labels_test)

    return round(accuracy, 2)


def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    """Trains an MLP classifier then returns the list of training accuracies and the final testing accuracy.

    Args:
        data_train (np.ndarray(np.float32)): The train dataset matrix of shape (train_size, d_in).
        labels_train (np.ndarray(np.float32)): The label vector of length train_size.
        data_test (np.ndarray(np.float32)): The test dataset matrix of shape (test_size, d_in).
        labels_test (np.ndarray(np.float32)): The test vector of length test_size.
        d_h (np.int64): The number of neurons in the hidden layer.
        learning_rate (np.float32): The learning rate of the network.
        num_epoch (np.int64): The number of steps used to train the Neural Network.

    Returns:
        train_accuracies (np.ndarray(np.float32)): The list of accuracy of the network while training.
        test_accuracy (np.float32): The final accuracy of the network on the test dataset.
    """

    d_in = np.shape(data_train)[1]
    d_out = 10

    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises

    train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)
    final_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)

    return train_accuracies, final_accuracy


if __name__ == "__main__":

    # Create train and test datasets
    data, labels = read_cifar("data")
    split_factor = 0.9
    a, b, c, d = split_dataset(data, labels, split_factor)

    # Define the network hyper-parameters and train it
    d_h = 64
    learning_rate = 0.1
    num_epoch = 100

    train_accuracies, final_accuracy = run_mlp_training(a, b, c, d, d_h, learning_rate, num_epoch)
    print("The accuracy of the network on the test dataset is " + str(final_accuracy) + "%")
    
    # Plot the evolution of the accuracy with the number training steps
    plt.plot(train_accuracies)
    plt.xlabel("Number of training steps")
    plt.ylabel("Accuracy (in %)")
    plt.title("Accuracy of the Neural Network depending on the number of iterations of training")
    plt.show()

