from network_data_preparation import NNData
from network_layers import LayerList
import math


class FFBPNetwork:
    """ Create a network based on LayerList to train and test data sets"""
    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        self._inputs = num_inputs
        self._outputs = num_outputs
        self.network = LayerList(self._inputs, self._outputs)

    def add_hidden_layer(self, num_nodes: int, position=0):
        """ Insert a hidden layer into the network """
        self.network.reset_to_head()
        if position < 0:
            raise IndexError
        if position == 0:
            self.network.add_layer(num_nodes=num_nodes)
        else:
            for i in range(0, position):
                self.network.move_forward()
            self.network.add_layer(num_nodes=num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2, order=NNData.Order.RANDOM):
        """ Train the network based on the training set """
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        for epoch in range(0, epochs):
            data_set.prime_data(order=order)
            sum_error = 0
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                for j, node in enumerate(self.network.input_nodes):
                    node.set_input(x[j])
                produced = []
                for j, node in enumerate(self.network.output_nodes):
                    node.set_expected(y[j])
                    sum_error += (node.value - y[j]) ** 2 / self._outputs
                    produced.append(node.value)
                if epoch % 1000 == 0 and verbosity > 1:
                    print("Sample", x, "expected", y, "produced", produced)
            if epoch % 100 == 0 and verbosity > 0:
                print("Epoch", epoch, "RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))
        print("Final Epoch RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        data_set.prime_data(order=order)
        sum_error = 0
        while not data_set.pool_is_empty(NNData.Set.TEST):
            x, y = data_set.get_one_item(NNData.Set.TEST)
            for j, node in enumerate(self.network.input_nodes):
                node.set_input(x[j])
            produced = []
            for j, node in enumerate(self.network.output_nodes):
                sum_error += (node.value - y[j]) ** 2
                produced.append(node.value)

            print(x, ",", y, ",", produced)
        print("RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TEST)))
