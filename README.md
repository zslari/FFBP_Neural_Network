# FFBP_Neural_Network
Feed Forward Back Propagating Neural Network built from scratch using Python

This repository holds files to demonstrate how an Artificial Neural Network (ANN) can be built with Python while maintaining Object Oriented Design principles.

The NNData class is designed to support methods that will allow us to efficiently manage our dataset, such as allowing us to split our dataset into training and testing data, prime our data, and then expose one set of features and labels to our neural network at a time. 

The MultilinkNode, Neurode, FFNeurode, BPNeurode, and FFBPNeurode classes make up the design for the individual neurons of our network and allow neurons to identify relationships with their neighboring downstream and upstream neurons and signal information onto those neurons. 

The LayerList class implements and holds the input, hidden, and output layers of the network from the LinkedList Abstract Data Type.

The FFBPNetwork class users LayerList to create a network and provides training and test methods to take advantage of the NNData class.

The graphic below from Buranajun, Prathana & Sasananan, Montalee & Sasananan, Setta. (2007). PREDICTION OF PRODUCT DESIGN AND DEVELOPMENT SUCCESS USING ARTIFICIAL NEURAL NETWORK shows how information from a single data point and its respective features are assigned to the input layer neurodes, while the labels, or expected values are assigned to the output layer neurodes.

The network then sends a signal downstream to hidden layer neurodes and finally to output layer neurodes comparing the results of what was predicted to what was expected. Once all neurodes have reported in, the backpropagation phase begins and feedback about the comparison of predicted versus expected results is sent upstream to input layer neurodes. This process usually needs to occur many, many, many, times to incrementally allow the network to move closer to correctly predicting the expected results.  After this process is performed on a portion of the trainining data set, the network will run with a testing data set and hopefully be able to predict a value close to our expected value with less and less error. 



![alt text](https://www.researchgate.net/profile/Montalee_Sasananan/publication/281271367/figure/fig2/AS:284441772609536@1444827611106/Feed-Forward-Neural-Network-with-Back-Propagation.png)







