# Image classification

Following [Quentin Gallouédec's tutorial](https://gitlab.ec-lyon.fr/qgalloue/image_classification_instructions), we implement 2 image classification algorithms in Python:
- k-nearest neighbors (KNN)
- artificial neural network (NN)

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is used to test these algorithms. 

## Description of the algorithms

### K-nearest neighbors (KNN)
K-nearest neighbors (KNN) algorithm is a classifier using the distance of an element of the test dataset from all of the train dataset. Here, we use the Euclidian L2 distance to compare 2 elements given their pixel information. 

Then, the labels of the k-nearest neighbors are grouped and counted and the most frequent one is selected as the predicted label. This allows to classify the test data in one of the categories of the train data. 

<figure>
<img src=https://www.jcchouinard.com/wp-content/uploads/2021/08/image-8.png.webp alt="Trulli" style="width:100%">
<figcaption align = "center"><b>KNN algorithm diagram</b></figcaption>
</figure>

_Source : https://www.jcchouinard.com/k-nearest-neighbors/_

### Artificial neural network (NN)
Artificial neural networks are Machine Learning algorithms and more specifically Deep Learning algorithms. The idea behind is biomimicry, and they reproduce the functionning of a human brain.

Neural networks implement layers of neurons connected to each other distributed in an input layer, hidden layers and an output layer. Each node connects with the nodes of the layers before and after its own, and a weight is attributed to each bond. Each node has a threshold value and an activation function, so that if the weighted inputs are greater than the threshold, it is activated and passes data according to its activation function (often chosen as a sigmoid function) to the next layer of the network.

<figure>
<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Neural_network_example.svg/1920px-Neural_network_example.svg.png alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Neural Network algorithm diagram</b></figcaption>
</figure>

_Source : https://en.wikipedia.org/wiki/Neural_network_

Artificial neural networks rely on iterations to find the optimum weights affected to each neuron connection. Once optimum meta-parameters are found, they are powerful tools that can by applied to a wide range of test datasets, considering they are similar enough in content. 

## Installation

### Requirements
This project requires [python3](https://www.python.org/), and common libraries installations :

- [NumPy](https://numpy.org/) for fast matrix computation
- [Matplotlib](https://matplotlib.org/) for result visualization
- [Pickle](https://docs.python.org/3/library/pickle.html) for Python object serialization

## Usage
Functions can be called by running each script in a terminal. An example of use is implemented in each script to show the process leading to a coherent and interpretable result of the algorithms.

## Support and contribution
If you'd like to suggest any corrections or improvements or need help understanding the algorithms or results, feel free to contribute by [opening an issue](https://gitlab.ec-lyon.fr/bcornill/image-classification/-/issues/new).

## Authors and acknowledgment
Implementation of the algorithms from the tutorial was made by [Barnabé Cornilleau](https://gitlab.ec-lyon.fr/bcornill) under the supervision of Emmanuel Dellandrea.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
