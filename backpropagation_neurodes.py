from layers import LayerType
from multi_link_node import MultiLinkNode
from neurode import Neurode
from feed_forward_neurode import FFNeurode


class BPNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value=None):
        """ Calculates the delta from expected values """
        weighted_sum = 0.0
        if self._node_type == LayerType.OUTPUT:
            self._delta = (expected_value - self._value) * self._sigmoid_derivative(self._value)
        else:
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                weighted_sum += node.delta * node.weight(self)
            self._delta = weighted_sum * self._sigmoid_derivative(self._value)

    def data_ready_downstream(self, node):
        """ Collects data from downstream nodes and makes data
         available to next layer
         """
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """ Allows client to directly set the value of an
        output layer neurode
        """
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node, adjustment):
        """ Adjust the importance of an upstream neurode """
        self._weights[node] += adjustment

    def _update_weights(self):
        """ Calculates the new weight/importance of a neurode """
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.adjust_weights(node=self,
                                adjustment=(node.delta *
                                            node.learning_rate * self._value))

    def _fire_upstream(self):
        """ Tells upstream neighbors data is available downstream """
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):
    pass
