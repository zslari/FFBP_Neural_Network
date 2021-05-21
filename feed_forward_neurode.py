import numpy as np
from multi_link_node import MultiLinkNode
from neurode import Neurode


class FFNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1/(1 + np.exp(-value))

    def _calculate_value(self):
        """ Calculates the weighted sum of the upstream neurodes values """
        sum_weighted_values = 0.0
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            sum_weighted_values += (self._weights[node] * node.value)
        self._value = self._sigmoid(sum_weighted_values)

    def _fire_downstream(self):
        """ Tells downstream neighbors data is available upstream """
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """ Collects data from upstream nodes and makes data
         available to next layer
         """
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """ Allows client to directly set the value of an
        input layer neurode
        """
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)
