# FFBP_Neural_Network
Feed Forward Back Propagating Neural Network built from scratch using Python

This repository holds files to demonstrate how an Artificial Neural Network can be built with Python while maintaining Object Oriented Design principles.

The graphic below shows how information from a single data point and its respective features are assigned to the input layer neurodes, while the labels, or expected values, are assigned to the output layer neurodes. 

The neural network sends a signal downstream to hidden layer neurodes and finally to output layer neurodes comparing the results of what was predicted to what was expected. Once all neurodes have reported in, the backpropagation phase begins and feedback about the comparison of predicted versus expected results is sent upstream to input layer neurodes. This process usually needs to occur many, many, many, times to incrementally allow the network to move closer to correctly predicting the expected results.

After this process is performed on a portion of the trainining data set, the network will run with a testing data set and hopefully be able to predict a value closer to our expected value with less and less error. 

![alt text](https://www.researchgate.net/profile/Montalee_Sasananan/publication/281271367/figure/fig2/AS:284441772609536@1444827611106/Feed-Forward-Neural-Network-with-Back-Propagation.png)


## Quick Start

1. Clone or download the repo
2. Run `main.py`
3. Choose to run the Iris.csv dataset, sin function, XOR with bias, or XOR without bias to test the network
