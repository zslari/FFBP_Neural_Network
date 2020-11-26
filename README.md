# FFBP_Neural_Network
Feed Forward Back Propagating Neural Network built from scratch using Python

This repository holds files to demonstrate how an Artificial Neural Network (ANN) can be built from scratch in Python while maintaining Object Oriented Design principles.

The NNData class is designed to support methods that will allow us to efficiently manage our dataset, such as allowing us to split our dataset into training and testing data, prime our data, and then expose one set of features and labels to our neural network at a time. 

The MultilinkNode, Neurode, FFNeurode, BPNeurode, and FFBPNeurode classes make up the design for the individual neurons of our network and allow neurons to identify relationships with their neighboring downstream and upstream neurons and signal information onto those neurons. 

The LayerList class implements and holds the input, hidden, and output layers of the network from the LinkedList Abstract Data Type.

The FFBPNetwork class users LayerList to create a network and provides training and test methods to take advantage of the NNData class.

The ANN is set to easily run with the XOR function (with and without a bias node), the value of the sin function from 0 to 1.57 radians, and the UCI Iris Dataset (CSV file found in this repo).







